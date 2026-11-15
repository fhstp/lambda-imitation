"""Report planned seeds, not jobs/checkpoints or favorable survivors."""

import numpy as np
import pytest

from .report import generate, interval, summarize_curves
from .records import write


def test_seed_interval_uses_student_t_and_rejects_missing_seed():
    result=interval([1,2,3,4,5])
    assert result["n_seeds"]==5 and result["mean"]==3
    assert result["ci95"][0]==pytest.approx(1.0367568,rel=1e-6)
    with pytest.raises(ValueError):interval([1,np.nan,3])


def curves():
    result={}
    for measure in ("uniform","discounted"):
        for correction in ("uncorrected","is","retrace"):
            offset=(1 if measure=="discounted" else 0)+(0.5 if correction=="retrace" else 0)
            result[(measure,correction)]={"coordinates":np.array([10,20]),
                "values":np.array([[1,2,3,4,5],[2,3,4,5,6]],float)+offset,
                "seeds":[1,2,3,4,5],"provenance":[]}
    return result


def test_paired_endpoint_contrasts_do_not_select_a_peak():
    result=summarize_curves(curves(),20)
    assert result["conditions"]["uniform/retrace"]["mean"]==4.5
    assert result["exploratory_paired_contrasts"]["retrace/discounted-minus-uniform"]["mean"]==1
    assert result["peak_selection"] is False
    assert result["multiple_testing_adjustment"] is False


def test_unmatched_seed_labels_and_missing_endpoint_are_rejected():
    changed=curves()
    changed[("uniform","is")]["seeds"]=[6,7,8,9,10]
    with pytest.raises(RuntimeError,match="unmatched paired contrast"):
        summarize_curves(changed,20)
    with pytest.raises(RuntimeError,match="endpoint missing"):
        summarize_curves(curves(),19)


def test_incomplete_study_does_not_produce_survivor_only_statistics(tmp_path):
    write(tmp_path/"study.json",{"jobs":[{"name":"failed-seed-block"}]})
    with pytest.raises(RuntimeError,match="refusing survivor-only"):
        generate(tmp_path)
    assert not (tmp_path/"report/summary.json").exists()
