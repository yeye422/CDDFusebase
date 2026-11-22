"""Synthetic regression tests; no real datasets, weights or batch evaluation."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diagnose_mars_frozen_prior as probe
from lwpd import HaarDWT2D
from mars_probe_evidence import (
    admission_decision, coherent_path_metrics, fit_projection, paired_evidence,
    project_rows, source_structure_features,
)


class EvidenceTests(unittest.TestCase):
    def test_projection_uses_only_training_statistics(self):
        rng=np.random.default_rng(1)
        train=rng.normal(size=(100,5))+np.arange(5)*3
        fitted=fit_projection(train,3)
        np.testing.assert_allclose(fitted["mean"],train.mean(0))
        projected=project_rows(fitted,train)
        np.testing.assert_allclose(projected.mean(0),0,atol=1e-6)
        before={key:value.copy() for key,value in fitted.items() if isinstance(value,np.ndarray)}
        project_rows(fitted,np.full((10,5),10000.0))
        for key,value in before.items():np.testing.assert_array_equal(fitted[key],value)

    def test_incremental_heads_find_missing_signal_and_match_control_capacity(self):
        rng=np.random.default_rng(2)
        branches=probe.method_branches([42,43,44])
        def rows(count):
            e2=rng.uniform(-1,1,(count,4)).astype(np.float32)
            missing=rng.uniform(-1,1,(count,1)).astype(np.float32)
            values={"e2_signed":e2[:,:2],"e2_rich":e2,"mars":missing,
                    "shifted":missing[rng.permutation(count)],"source":np.abs(e2[:,:1])}
            values.update({f"random_{seed}":rng.uniform(-1,1,(count,1)).astype(np.float32)
                           for seed in (42,43,44)})
            return values,0.5+0.1*e2[:,0]+0.2*missing[:,0]
        sampled,target=rows(512)
        heads,projections=probe.fit_incremental_heads(sampled,target,branches,1)
        self.assertIs(projections["mars"],projections["shifted"])
        self.assertEqual({len(heads[name]["coef"]) for name,branch in branches.items()
                          if branch is not None},{5})
        validation,truth=rows(256)
        maps={name:value.T.reshape(value.shape[1],16,16) for name,value in validation.items()}
        predictions=probe.predict_incremental_heads(maps,branches,heads,projections)
        for method,branch in branches.items():
            if branch is None:continue
            fitted=heads[method]["residualizer"]
            projected=project_rows(projections[branch],validation[branch])
            residual=(projected-fitted["branch_mean"]
                      -((validation["e2_rich"]-fitted["mean"])/fitted["std"])@fitted["coef"])
            explicit=np.concatenate((validation["e2_rich"],residual),axis=1)
            expected=probe.predict_ridge(heads[method],explicit.T[:,:,None]).reshape(16,16)
            np.testing.assert_allclose(predictions[method],expected,atol=1e-6)
        errors={name:float(np.mean((value.ravel()-truth)**2)) for name,value in predictions.items()}
        self.assertLess(errors["e2_plus_mars"],errors["e2_rich"]*0.01)
        for seed in (42,43,44):
            self.assertLess(errors["e2_plus_mars"],errors[f"e2_plus_random_{seed}"]*0.01)

    def test_redundant_features_cannot_improve_by_reducing_ridge_shrinkage(self):
        rng=np.random.default_rng(7)
        base=rng.uniform(-1,1,(512,4)).astype(np.float32)
        branches=probe.method_branches([42])
        sampled={"e2_rich":base,"e2_signed":base[:,:2],
                 **{name:base[:,:1].copy() for name in ("mars","shifted","source","random_42")}}
        heads,projections=probe.fit_incremental_heads(sampled,0.5+0.2*base[:,0],branches,1)
        val=rng.uniform(-1,1,(256,4)).astype(np.float32)
        rows={"e2_rich":val,"e2_signed":val[:,:2],
              **{name:val[:,:1].copy() for name in ("mars","shifted","source","random_42")}}
        maps={name:value.T.reshape(value.shape[1],16,16) for name,value in rows.items()}
        predictions=probe.predict_incremental_heads(maps,branches,heads,projections)
        np.testing.assert_allclose(predictions["e2_plus_mars"],predictions["e2_rich"],atol=1e-6)

    def test_e2_magnitude_is_computed_before_signed_pooling_cancels(self):
        haar=HaarDWT2D()
        high=torch.ones((1,1,128,128));high[:,:,:,1::2]=-1
        zero=torch.zeros_like(high)
        detail=haar.inverse((zero,high,zero,zero),(256,256))
        shared=torch.ones_like(detail)
        model=SimpleNamespace(lwpd=SimpleNamespace(haar=haar),
            encoder=SimpleNamespace(patch_embed=lambda x:shared,encoder_level1=lambda x:x),
            encode_pair=lambda optical,sar:(shared,detail,shared,detail))
        image=np.zeros((256,256),np.uint8)
        with patch.object(probe,"read_pair",return_value=(image,image)):
            encoded=probe.extract_e2(model,("o","s"),torch.device("cpu"))
        self.assertLess(float(np.abs(encoded["e2_detail"][1]).max()),1e-6)
        np.testing.assert_allclose(encoded["e2_absolute"][1][0],1,atol=1e-6)
        np.testing.assert_allclose(encoded["e2_energy"][1],1,atol=1e-6)

    def test_source_structure_control_is_finite_32_channels(self):
        image=np.zeros((256,256),np.uint8);image[:,128:]=255
        for size in (128,64):
            values=source_structure_features(image,size)
            self.assertEqual(values.shape,(32,size,size))
            self.assertTrue(np.isfinite(values).all())
            self.assertGreater(float(values.std()),0.01)

    def test_path_rejects_teleporting_corridor_maxima(self):
        corridor=np.zeros((9,33))
        corridor[0,0::2]=1;corridor[8,1::2]=1
        self.assertTrue((corridor.max(0)==1).all())
        score=coherent_path_metrics(corridor)
        self.assertLess(score["path_coverage"],0.6)
        self.assertLess(score["longest_run_fraction"],0.1)

    def test_path_reports_longest_run_and_max_gap_separately(self):
        corridor=np.array([[1]*10+[0]*13+[1]*10],np.float32)
        score=coherent_path_metrics(corridor)
        self.assertAlmostEqual(score["path_coverage"],20/33)
        self.assertAlmostEqual(score["longest_run_fraction"],10/33)
        self.assertAlmostEqual(score["max_gap_fraction"],13/33)

    def test_all_methods_share_finite_scene_intersection(self):
        rows=[]
        for scene in ("a","b","c"):
            for method,value in (("mars",1.0),("base",2.0),("random",3.0)):
                rows.append(dict(stage=0,cohort="unaudited",scene=scene,method=method,
                                 mse=float("nan") if scene=="b" and method=="random" else value))
        pairs=paired_evidence(rows,["mars","base","random"],"mars",["base","random"],
                              {"mse":False},bootstrap=32)
        for row in pairs:
            self.assertEqual(row["scene_ids"],["a","c"])
            self.assertEqual(row["n_scenes"],2)
            self.assertGreater(row["ci95_low"],0)

    def test_line_bootstrap_clusters_by_scene_and_negative_direction_is_reversed(self):
        rows=[]
        for scene in ("a","b"):
            for label in range(3):
                for method,value in (("mars",0.1),("base",0.3)):
                    rows.append(dict(scene=scene,id=label,stage=0,cohort="disconnected",
                                     method=method,path_coverage=value))
        row=paired_evidence(rows,["mars","base"],"mars",["base"],
                            {"path_coverage":True},bootstrap=32)[0]
        self.assertEqual(row["n_scenes"],2)
        self.assertEqual(row["n_items"],6)
        self.assertAlmostEqual(row["mean_gain"],0.2)

    @staticmethod
    def positive_evidence():
        random=[f"e2_plus_random_{seed}" for seed in (42,43,44)]
        pixels=[];lines=[]
        for stage in (0,1):
            for reference in ["e2_rich","e2_plus_shifted"]+random:
                for metric in ("mse","high_edge_mae"):
                    pixels.append(dict(stage=stage,cohort="unaudited",reference=reference,
                        metric=metric,n_scenes=56,n_items=56,relative_gain=0.1,
                        mean_gain=0.01,median_gain=0.01,ci95_low=0.005,wins=40))
            for reference in ("e2_rich","e2_plus_source"):
                for cohort,count in (("missing",3),("disconnected",16)):
                    for metric in ("longest_run_fraction","max_gap_fraction","path_coverage"):
                        lines.append(dict(stage=stage,cohort=cohort,reference=reference,
                            metric=metric,n_scenes=3 if cohort=="missing" else 8,
                            n_items=count,mean_gain=0.1,ci95_low=0.05,wins=3))
        return pixels,lines,random

    def test_decision_requires_both_stages_all_seeds_and_safe_negatives(self):
        pixels,lines,random=self.positive_evidence()
        decision=admission_decision(pixels,lines,random,32,"gray")
        self.assertEqual(decision["status"],"candidate_for_small_adapter_ablation")
        self.assertEqual(admission_decision(pixels,lines,random,8,"gray")["status"],"insufficient_data")
        self.assertEqual(admission_decision(pixels,lines,random[:1],32,"gray")["status"],"insufficient_data")
        unsafe=copy.deepcopy(lines)
        for row in unsafe:
            if row["cohort"]=="disconnected":row["ci95_low"]=-0.1
        self.assertEqual(admission_decision(pixels,unsafe,random,32,"gray")["status"],
                         "inconclusive_structure_or_stage")
        weak=copy.deepcopy(pixels)
        for row in weak:
            if row["reference"]==random[-1]:row["ci95_low"]=-0.01
        self.assertEqual(admission_decision(weak,lines,random,32,"gray")["status"],
                         "not_supported_for_current_probe_route")
        self.assertEqual(admission_decision(pixels,lines,random,32,"rgb")["status"],
                         "exploratory_rgb_requires_gray_confirmation")

    def test_validation_selection_excludes_train_and_keeps_all_audited(self):
        manifest={"splits":{"train":["t"],"val":[str(i) for i in range(20)],"test":["x"]}}
        audit=["1","4","7","10","12","14","17","19"]
        selected=probe.validation_ids(manifest,audit,12)
        self.assertEqual(len(selected),12)
        self.assertTrue(set(audit).issubset(selected))
        self.assertFalse(set(selected)&{"t","x"})
        with self.assertRaises(ValueError):probe.validation_ids(manifest,audit,7)
        manifest["splits"]["train"]=["1"]
        with self.assertRaises(ValueError):probe.validation_ids(manifest,audit,12)

    def test_synthetic_run_saves_protocol_raw_evidence_and_refuses_overwrite(self):
        # Exercise the entire orchestration with tiny synthetic feature maps.
        # No real image, E2/MaRS weights or official evaluation is accessed.
        with TemporaryDirectory() as folder:
            root=Path(folder).resolve();labels_dir=root/"labels";labels_dir.mkdir()
            for name in ("source_labels.csv","e2_visibility.csv"):
                (labels_dir/name).write_text("synthetic",encoding="utf-8")
            (root/"mars_probe_evidence.py").write_text("synthetic helper",encoding="utf-8")
            torch.save({"model":{}},root/"e2.pth")
            (root/"mars.pth").write_bytes(b"synthetic weight")
            val=[str(i) for i in range(10,20)]
            manifest={"optical_dir":"o","sar_dir":"s",
                      "splits":{"train":["1","2"],"val":val,"test":["99"]}}
            (root/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
            archive={"checkpoints":{"e2":"e2.pth"},
                     "checkpoint_hashes":{"e2":probe.sha256_file(root/"e2.pth")}}
            audit={scene:() for scene in val[:8]}
            labels=[dict(scene="10",id="a",label="connected",x1="20",y1="20",x2="80",y2="20"),
                    dict(scene="10",id="b",label="disconnected",x1="20",y1="40",x2="80",y2="40")]
            def synthetic_features(paths,models,device,mode,runtime):
                rng=np.random.default_rng(int(paths[0].stem))
                features={}
                for stage,size in ((0,8),(1,4)):
                    base=rng.normal(size=(8,size,size)).astype(np.float32)
                    features[stage]={"e2_signed":base[:4],"e2_rich":base,
                        **{name:rng.normal(size=(8,size,size)).astype(np.float32)
                           for name in ("mars","shifted","source","random_42")}}
                return features,rng.uniform(0,1,(256,256)).astype(np.float32)
            args=SimpleNamespace(train_scenes=2,val_scenes=10,samples_per_scene=32,
                augmentation_dim=2,random_seeds=[42],save_images=0,input_mode="gray",
                cpu=True,mars_weights=Path("mars.pth"),output_dir=Path("result/run"))
            original_cwd=Path.cwd()
            try:
                with patch.object(probe,"ROOT",root),patch.object(probe,"DEFAULT_LABELS",labels_dir), \
                     patch.object(probe,"GRIDS",{0:8,1:4}), \
                     patch.object(probe,"load_inputs",return_value=(archive,{"source_manifest":"manifest.json"},labels,audit)), \
                     patch.object(probe,"load_visibility",return_value={("10","a"):"missing"}), \
                     patch.object(probe,"OpticalBaseAnchoredE2",side_effect=torch.nn.Identity), \
                     patch.object(probe,"load_mars",side_effect=lambda *args:torch.nn.Identity()), \
                     patch.object(probe,"scene_features",side_effect=synthetic_features), \
                     patch.object(probe,"register") as register, \
                     contextlib.redirect_stdout(io.StringIO()):
                    probe.run(args)
                    out=root/"result/run"
                    summary=json.loads((out/"summary.json").read_text(encoding="utf-8"))
                    meta=json.loads((out/"run_metadata.json").read_text(encoding="utf-8"))
                    self.assertEqual(summary["decision"]["status"],"insufficient_data")
                    self.assertEqual(meta["metric_profile"],"mars_incremental_structure_screen_v3")
                    self.assertEqual(len(meta["unaudited_val_ids"]),2)
                    self.assertTrue((out/"readout_and_projection.npz").is_file())
                    self.assertTrue((out/"接入价值筛查.md").is_file())
                    with (out/"pixel_probe.csv").open(encoding="utf-8-sig") as handle:
                        self.assertEqual(len(list(csv.DictReader(handle))),120)
                    register.assert_called_once()
                    with self.assertRaises(FileExistsError):probe.run(args)
            finally:
                import os
                os.chdir(original_cwd)


if __name__=="__main__":
    unittest.main()
