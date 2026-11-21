"""Reconstruct frozen candidate evidence without fitting images or changing cuts."""
import numpy as np

PARAMETERS=('x','y','qx','qy','inverse_distance')
CASES={'physical':(1.,1.),'negative':(-1.,1.),'reversed':(1.,-1.)}


def boolean(value):
    if isinstance(value,(bool,np.bool_)):return bool(value)
    if value=='True':return True
    if value=='False':return False
    raise ValueError('Expected explicit boolean value')


def integer(value):
    number=float(value)
    if not np.isfinite(number) or int(number)!=number:raise ValueError('Expected finite integer')
    return int(number)


def likelihood(flux,sigma,sign):
    f=np.asarray(flux,dtype=np.longdouble);s=np.asarray(sigma,dtype=np.longdouble)
    if not len(f) or f.shape!=s.shape or not np.isfinite(np.r_[f,s]).all() or np.any(s<=0):
        raise ValueError('Invalid independent-sector scalar likelihood')
    weights=s**-2;total=weights.sum();amplitude=np.sum(f*weights)/total;error=1/np.sqrt(total)
    z=sign*amplitude/error;chi=np.sum(weights*(f-amplitude)**2)
    return dict(flux=float(amplitude),sigma=float(error),formal_snr=float(z),
        sector_consistency_chi2=float(chi),coherent_score=float(z-np.sqrt(chi))),weights


def review_case(candidates,measurements,validity,sectors,case,control_maximum):
    """All admitted paths, including partial-sector likelihood and exact duplicates.

    Leaving out a sector is a diagnostic only. Its reduced anchor count is
    reported explicitly; no omitted-sector statistic is used as a new cut.
    """
    if case not in CASES or not np.isfinite(control_maximum):raise ValueError('Invalid arm or control threshold')
    if len(set(sectors))!=len(sectors):raise ValueError('Duplicate declared sector')
    sign,orientation=CASES[case];allowed=set(sectors);seen=set();accepted=set()
    for row in validity:
        key=integer(row['proposal_rank'])
        if key<1 or key in seen:raise ValueError('Duplicate or invalid proposal validity key')
        seen.add(key)
        if boolean(row['native_valid']):accepted.add(key)
    groups={};pairs=set()
    for row in measurements:
        proposal,sector=integer(row['proposal_rank']),integer(row['sector'])
        if proposal not in seen or sector not in allowed:raise ValueError('Undeclared photometry key')
        if (proposal,sector) in pairs:raise ValueError('Duplicate sector evidence')
        pairs.add((proposal,sector));groups.setdefault(proposal,[]).append(row)
    candidate_ids=[integer(row['proposal_rank']) for row in candidates]
    if len(set(candidate_ids))!=len(candidate_ids) or set(candidate_ids)!=accepted:
        raise ValueError('Admitted candidates disagree with complete validity table')
    results=[];omissions=[];maximum_error=0.;unique=set();above_unique=set()
    for index,row in enumerate(candidates,1):
        proposal=integer(row['proposal_rank']);rank=integer(row['coherent_rank'])
        if rank!=index:raise ValueError('Candidate rank order differs from frozen table')
        if row.get('case')!=case or float(row['sign'])!=sign or float(row['orientation'])!=orientation:
            raise ValueError('Physical/control arm identity differs')
        parameters=tuple(float(row[k]) for k in PARAMETERS)
        if not np.isfinite(parameters).all() or parameters[-1]<=0:raise ValueError('Invalid trajectory parameters')
        np.testing.assert_allclose(float(row['trial_distance_au']),1/parameters[-1],rtol=1e-10,atol=1e-10)
        part=sorted(groups.get(proposal,[]),key=lambda r:integer(r['sector']))
        anchors=np.array([boolean(r['anchor']) for r in part]);n=len(part)
        if n!=integer(row['sectors']) or anchors.sum()<3 or anchors.sum()!=integer(row['anchors']):
            raise ValueError('Missing sector evidence or anchor gate mismatch')
        if n-anchors.sum()!=integer(row['partial_sectors']):raise ValueError('Partial sector count differs')
        sid=np.array([integer(r['sector']) for r in part]);f=np.array([float(r['flux']) for r in part]);s=np.array([float(r['sigma']) for r in part])
        for r in part:
            coverage,overlap=float(r['coverage']),float(r['stamp_overlap_fraction'])
            if not np.isfinite([coverage,overlap]).all() or not 0<=coverage<=overlap<=1 or boolean(r['anchor'])!=(coverage>=.95):
                raise ValueError('Invalid native footprint or anchor classification')
        result,weights=likelihood(f,s,sign)
        expected=np.array([float(row[k]) for k in result]);actual=np.array(list(result.values()))
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose([float(r['formal_snr']) for r in part],sign*f/s,rtol=1e-10,atol=1e-10)
        if integer(row['positive_sectors'])!=int(np.sum(sign*f>0)):raise ValueError('Positive-sector accounting differs')
        np.testing.assert_allclose(float(row['minimum_sector_score']),np.min(sign*f/s),rtol=1e-10,atol=1e-10)
        maximum_error=max(maximum_error,float(np.max(abs(actual-expected))))
        dominant=int(np.argmax(weights))
        # Preserve the exact saved statistic at the frozen strict threshold.
        above=float(row['coherent_score'])>control_maximum
        result={key:float(row[key]) for key in result}
        unique.add(parameters)
        if above:above_unique.add(parameters)
        record=dict(case=case,coherent_rank=rank,proposal_rank=proposal,
            **{k:float(row[k]) for k in PARAMETERS},trial_distance_au=float(row['trial_distance_au']),
            **result,sectors=n,anchors=int(anchors.sum()),partial_sectors=int(n-anchors.sum()),
            included_sectors=';'.join(map(str,sid)),not_included_sectors=';'.join(map(str,sorted(allowed-set(sid)))),
            positive_sectors=int(np.sum(sign*f>0)),dominant_sector=int(sid[dominant]),
            dominant_information_fraction=float(weights[dominant]/weights.sum()),
            above_observed_control=above,control_maximum=float(control_maximum))
        for i,sector in enumerate(sid):
            keep=np.arange(n)!=i;omitted,_=likelihood(f[keep],s[keep],sign)
            tension=sign*(f[i]-omitted['flux'])/np.hypot(s[i],omitted['sigma'])
            entry=dict(case=case,proposal_rank=proposal,omitted_sector=int(sector),
                anchors_remaining=int(anchors[keep].sum()),meets_original_anchor_gate=bool(anchors[keep].sum()>=3),
                **omitted,signed_flux_tension=float(tension))
            omissions.append(entry)
            if i==dominant:record.update(without_dominant_signed_snr=omitted['formal_snr'],
                without_dominant_coherent_score=omitted['coherent_score'],
                without_dominant_anchors=entry['anchors_remaining'],dominant_signed_flux_tension=float(tension))
        results.append(record)
    if any(a['coherent_score']<b['coherent_score']-1e-10 for a,b in zip(results,results[1:])):
        raise ValueError('Frozen candidate order is not descending by score')
    summary=dict(case=case,admitted_proposals=len(results),total_proposals=len(seen),
        unique_parameter_tuples=len(unique),above_observed_control=sum(r['above_observed_control'] for r in results),
        above_control_unique_parameter_tuples=len(above_unique),maximum_reproduction_error=maximum_error,
        largest_coherent_score=max((r['coherent_score'] for r in results),default=None),
        top20_dominated_above90percent=sum(r['dominant_information_fraction']>.9 for r in results[:20]))
    return results,omissions,summary
