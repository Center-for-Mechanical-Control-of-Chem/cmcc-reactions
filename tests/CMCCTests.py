import os, sys

import numpy as np

dev_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
root = os.path.join(dev_root, 'cmcc-reactions')
deps = os.path.join(dev_root, 'dependencies')
sys.path.insert(0, root)
sys.path.insert(0, deps)
# os.chdir(root)

import os.path
import unittest
import itertools
import pprint
import tempfile as tf
import cmcc_reactions.reaction_data_schema as schema
# import cmcc_reactions.data_analysis_tools as thc_tools
import cmcc_reactions.trajectory_tools as trajt
import cmcc_reactions.reaction_data_analysis as rda
import cmcc_reactions.utils as utils
import cmcc_reactions.pipeline as pipeline
import cmcc_reactions.generate_reaction_products as gen_prods
import cmcc_reactions.coordinate_choice as cocho
import cmcc_reactions.optimal_directions as fopt

__all__ = [
    "CMCCTests"
]


def test_data(*path):
    return os.path.join(root, "tests", "TestData", *path)
class CMCCTests(unittest.TestCase):

    @unittest.skip
    def test_DataSerializer(self):
        self.skipTest('...')
        tree = {
            'a':{
                'b':{'c':np.array([[1, 2, 3], [4, 5, 6]]), 'e':[3, 4, 5]},
                'd':{'c':np.array([1, 2])},
                'c':10
            },
            'f':{'g':[1], 'c':[6, 7], 'x':np.random.rand(100, 100)}
        }
        serialized = schema.compress_tree(tree)
        # pprint.pprint(serialized)
        deserialized = schema.decompress_tree(serialized)
        # pprint.pprint(deserialized)


        with tf.TemporaryDirectory() as td:
            js_file = os.path.join(td, 'tree.json')
            schema.write_tree(js_file, tree, mode='json')

            tmp_file = os.path.join(td, 'tree.npz')
            schema.write_tree(tmp_file, tree)

            tree2 = schema.read_tree(tmp_file)
            tree3 = schema.read_tree(js_file, mode='json')
            # pprint.pprint(tree2)
            # pprint.pprint(tree3)

            print(os.path.getsize(tmp_file))
            print(os.path.getsize(js_file))

    @unittest.skip
    def test_DataValidator(self):
        data = {
            'reaction_1': {
                'system_1' : {
                    'faked': {
                        'atoms':['C'] * 10,
                        'solvothermal': {
                            "results": [
                                {
                                    'reactant':{
                                        'coordinates': np.random.rand(10, 3),
                                        'hessian': np.random.rand(30, 30),
                                    },

                                    'transition_state': {
                                        'coordinates': np.random.rand(10, 3),
                                        'hessian': np.random.rand(30, 30)
                                    }
                                }

                            ]
                        },

                        'internal_0': {
                            'spec':[1, 3, 2],
                            "results": [
                                {
                                    'reactant': {
                                        'coordinates': np.random.rand(10, 3)
                                    },

                                    'transition_state': {
                                        'coordinates': np.random.rand(10, 3)
                                    }
                                }

                            ]
                        }
                    }
                }
            }
        }

        ugh = schema.compress_tree(data)

        # huh = schema.validate_distortion_data(data)

        with tf.TemporaryDirectory() as td:
            js_file = os.path.join(td, 'tree.json')
            schema.write_distortion_data(js_file, data, mode='json')
            new_data = schema.read_distortion_data(js_file, data, mode='json')

            np_file = os.path.join(td, 'tree.npz')
            schema.write_distortion_data(np_file, data)
            new_data = schema.read_distortion_data(np_file, data)

    @unittest.skip
    def test_CompileReactionSets(self):
        thc_res_loc = os.path.expanduser('~/Documents/Postdoc/Projects/CMCC/DA_res')
        os.chdir(thc_res_loc)
        for k in range(4, 6):
            comp = thc_tools.compile_reaction_class(f'Results_{k+1}')
            thc_tools.write_aggregate_data(f'Res{k+1}_aggregate.npz', comp)
            remp = thc_tools.load_aggregate_data(f'Res{k+1}_aggregate.npz')

        # print(comp.keys())
        # print(comp['coords'][0].shape)
        # for r,c in zip(remp['coords'], comp['coords']):
        #     print(r-c)

    @unittest.skip
    def test_GenerateReactionProducts(self):
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        dienes = [
            ["CH", "CH", "CH", "CH"],
            # ["C@@H", "C", "C@H", "C@H"],
            # ["C@@H", "C", "C@H", "C@@H"],
            # ["C@H", "C", "C@@H", "C@H"]
        ]
        diene_template = "[{D[0]}:1]([CH2:7]2)[{D[1]}:5]=[{D[2]}:6][{D[3]}:2]2"
        base_templates = [
            "[R][C@@H:3]1[diene][C@@H:4]1[X]",
            "[R][C@@H:3]1[diene][C@H:4]1[X]"
        ]
        replacements = {
            'R': ["O=S(=O)(C2=CC=CC=C2)", "NC", "NC(=O)", "FC(C=C3)=CC=C3"],
            'X': ["S(=O)(C4=CC=CC=C4)=O", "CN", "C(=O)N", "C5=CC=C(F)C=C5"],
            "R'": ["S(=O)(C6=CC=CC=C6)=O", "CN", "C(=O)N", "C7=CC=C(F)C=C7"]
        }

        import shutil
        output_dir = os.path.expanduser('~/Desktop/test_smi')
        shutil.rmtree(output_dir)
        dat = gen_prods.generate_products_and_optimize(
            base_templates[:1],
            dienes,
            {
                k: v
                for k, v in replacements.items()
            },
            diene_template=diene_template,
            evaluate_energy=False,
            preoptimize=False,
            output_dir=output_dir,
            parallelizer=None,
            verbose=True
        )
        print(dat)

        # mod_smi = gen_prods.modify_template(gen_prods.template_1, {'R':'CCC', 'X':'ONO', "R'":'F'})
        # mod_smi = gen_prods.modify_template(gen_prods.template_5, {'R':'CCC', 'X':'ONO', "R'":'F'})
        # print(
        #     mod_smi,
        #     gen_prods.bond_breaking_indices(mod_smi)
        # )

    @unittest.skip
    def test_InitialSampling(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        init_data = utils.read_namedtuple(
            test_data('problem_product.json'),
            nt_type='InitialProductData'
        )

        presamp = gen_prods.generate_reactants_from_products(
            init_data,
            max_iterations=0,
            # output_dir=test_data()
        )

        return

        init_data = utils.read_namedtuple(
            test_data('product.json'),
            nt_type='InitialProductData'
        )

        presamp = gen_prods.generate_reactants_from_products(
            init_data,
            max_iterations=5,
            output_dir=test_data()
        )

    @unittest.skip
    def test_RefinedSampling(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # init_data = utils.read_namedtuple(
        #     test_data('trajectory.json'),
        #     # nt_type='InitialProductData'
        # )
        traj_data = utils.read_namedtuple(
            test_data('trajectory_problem.json'),
            # nt_type='ReoptimizedTrajectoryData'
        )
        traj_data = utils.read_namedtuple(
            test_data('trajectory.json'),
            # nt_type='ReoptimizedTrajectoryData'
        )

        new_traj = gen_prods.refine_trajectory(
            traj_data,
            profile_generator='pys-gsm',
            coord_type='cartesian',
            # num_images=30,
            # max_iterations=5,
            # param='energy',
            # optimizer='lbfgs'
            # ts_opt_generator=None
            # ts_opt_settings={}
        )

        rda.compare_profiles(new_traj, marker='.').show()

    @unittest.skip
    def test_LocalizedOptimization(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        traj_file = test_data('profile_problem.json')
        base_file = test_data('trajectory_problem.json')
        reoptimize = False
        use_original = False
        if (reoptimize and use_original) or not os.path.exists(traj_file):
            pre_string = utils.read_namedtuple(base_file)
            new_traj = gen_prods.refine_trajectory(pre_string,
                                                   refine_endpoints=True,
                                                   optimizer_settings=dict(thresh='gau_tight'))
            # new_traj = gen_prods.refine_trajectory(pre_string)
            ref1 = rda.DielsAlderReactionTrajectory.from_trajectory_data(new_traj)
            ref1.save(traj_file)
        elif reoptimize:
            pre_string = utils.read_namedtuple(traj_file)
            new_traj = gen_prods.refine_trajectory(pre_string,
                                                   refine_endpoints=True,
                                                   optimizer_settings=dict(thresh='gau_tight'))
            # new_traj = gen_prods.refine_trajectory(pre_string)
            ref1 = rda.DielsAlderReactionTrajectory.from_trajectory_data(new_traj)
            ref1.save(traj_file)
        else:
            ref1 = rda.DielsAlderReactionTrajectory.from_file(traj_file)
        opt = fopt.ForceOptimizer(ref1.reactant, ref1.transition_state, fragment_indices=1)
        opt.animate_normed(0, backend='x3d', mag=2).show()

    @unittest.skip
    def test_AdjustedForceOptimizer(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        import McUtils.Coordinerds as coordops

        ref1 = rda.DielsAlderReactionTrajectory.from_file(test_data('trajectory_problem.json'))

        zm_coords = [
            coordops.extract_zmatrix_internals(z)
            for z in ref1.reactant.get_bond_zmatrix(connect_fragments=False,
                                                    fragment_ordering=[0, 1],
                                                    attachment_points={0: 2}
                                                    )
        ]
        all_ints = sum(zm_coords, [])
        zm_full = ref1.reactant.get_bond_zmatrix(
            connect_fragments=True,
            fragment_ordering=[0, 1],
            attachment_points={0: 2}
        )

        opt = fopt.ForceOptimizer(ref1.reactant, ref1.transition_state,
                                  # fragment_indices=list(range(len(ref1.reactant.atoms)))[4:],
                                  # fragment_indices=1,
                                  fragment_indices=ref1.reactant.fragment_indices[1][3:],
                                  remove_fragment_transrot=False,
                                  remove_local_transrot=True,
                                  allow_mode_mixing=True,
                                  projection_internals=zm_coords[1],
                                  internals=zm_full
                                  # project_internals=False,
                                  # use_mode_space=False
                                  )

        opt.save(test_data('optimized_forces.json'))

    @unittest.skip
    def test_ForceAdjustedProfile(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        from McUtils.Data import UnitsData

        opt = fopt.ForceOptimizer.from_file(test_data('optimized_forces.json'))

        r, ts, fmrd = opt.reoptimize_with_force(0,
                                                -100,
                                                units='PicoJoules/Meters',
                                                # units=None,
                                                # optimizer_mode='pysis',
                                                # optimizer_method='lbfgs',
                                                # profile_generator='pys-dimer',
                                                # profile_generator='relaxed',
                                                # profile_generator='pys-cos',
                                                # optimizer_mode='ase',
                                                # optimizer_method='bfgs',
                                                profile_generator='ase-dimer',
                                                # apply_constraints=False,
                                                # modify_forces=True,
                                                # optimizer_mode='default',
                                                # optimizer_method='quasi-newton',
                                                optimizer_mode='scipy',
                                                optimizer_method='bfgs',
                                                apply_constraints=True,
                                                use_internals=False,
                                                mass_weight=False,
                                                reoptimize_reactants=True,
                                                reoptimize_ts=False,
                                                initial_ts_step=1,
                                                initial_reactants_step=1,
                                                # num_ts_steps=3,
                                                max_iterations=1000,
                                                max_displacement=.05,
                                                track_best=False)
        fmrd: fopt.ForceModifiedReactionData

        rf, tf, r0, t0 = [
            e * UnitsData.convert("Hartrees", "Kilocalories/Mole")
            for e in [
                fmrd.force_modified_reactant_energy,
                fmrd.force_modified_transition_state_energy,
                fmrd.reactant_energy,
                fmrd.transition_state_energy
            ]
        ]
        # print([
        #     rf, tf, r0, t0,
        # ])
        print(rf - r0)
        print(tf - t0)

        scan = r.embed_coords(
            [fmrd.reactant_geom, fmrd.force_modified_reactant_geom],
            sel=r.fragment_indices[0]
        )
        r.plot(scan, include_save_buttons=True, background='white').show()

        scan = ts.embed_coords(
            [fmrd.transition_state_geom, fmrd.force_modified_transition_state_geom],
            sel=r.fragment_indices[0]
        )
        ts.plot(scan, include_save_buttons=True, background='white').show()

    @unittest.skip
    def test_ForceOptimizerPipeline(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        uuh = pipeline.run_optimization_pipeline(
            test_data('product.json'),
            output_file=test_data('pipline_output.json'),
            max_iterations=5
        )
        print(uuh)

    @unittest.skip
    def test_CoordinateSystemGen(self):
        from Psience.Molecools import Molecule

        ts_samp = Molecule.from_file(test_data('ts_samp.xyz'))
        cocho.get_mostly_fixed_coordinate_system(ts_samp, [(22, 18, 19, 20)])

    @unittest.skip
    def test_CompressFSTree(self):
        tree = utils.construct_json_file_tree(test_data('test_smi'))
        import pprint
        pprint.pprint(tree)

    @unittest.skip
    def test_DA_Analysis(self):
        rda.DielsAlderReactionTrajectory.from_file(
            test_data('refined.json')
        ).plot_profile(
            distance_metric=rda.incremental_rmsds
        ).show()

    @unittest.skip
    def test_NewBuildingBlocks(self):
        # from Psience.Molecools import Molecule

        base_template = '[CH2:3]1[CH:1]([CH2:7]2)[CH:5]=[CH:6][CH:2]2[CH2:4]1'

        temp = base_template
        # Molecule.from_string(base_template).plot().show()
        # Molecule.from_string("[CH3:1]N").plot().show()

        temp = gen_prods.join_fragments(temp,
                                        "[CH3:1]N",
                                        [[2, 0]]
                                        )
        temp = gen_prods.join_fragments(temp,
                                        "[SH:1](=O)(C4=CC=CC=C4)=O",
                                        [[3, 0]]
                                        )
        temp = gen_prods.set_chiralities(temp, {3:'cw', 4:'cw'})
        print(temp)
        # Molecule.from_string(temp).plot().show()
        # print(
        #     gen_prods.join_fragments(
        #         '[C@@H:3]1[CH:1]([CH2:7]2)[CH:5]=[CH:6][CH:2]2[C@@H:4]1',
        #         "N[C:1]",
        #         [[2, 0]]
        #     )
        # )

    @unittest.skip
    def test_NewEnumeration(self):
        from Psience.Molecools import Molecule

        for n,smi in enumerate(gen_prods.fragment_to_smiles_iterator(
            '[C:3]1[C:1]([C:7]2)[C:5]=[C:6][C:2]2[C:4]1',
            ["[C:1]N", "[S:1](=O)(C4=CC=CC=C4)=O"],
            [2, 3],
            chiralities=[['cw', 'ccw'], ['cw', 'ccw']]
        )):
            Molecule.from_string(smi).plot(highlight_atoms=[0, 1]).show()
            if n >= 4:
                break

    @unittest.skip
    def test_ReoptZMIssues(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # raise Exception(
        #     gen_prods.inchi_key('N[CH2:8][C@H:3]1[CH:1]2[CH:5]=[CH:6][C@@H:2]([CH:4]1[S:9](=O)(=O)c1ccccc1)[CH2:7]2')
        # )

        uuh = pipeline.run_optimization_pipeline(
            test_data('problem_product2.json'),
            output_file=test_data('problem_pipline_output2.json'),
            max_iterations=5
        )
        print(uuh)

    @unittest.skip
    def test_ResultsAnalysis(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        res = pipeline.OptimizedForceResults.from_file(test_data('pipeline_data3.json'))
        # res = pipeline.run_optimization_pipeline(
        #     test_data('pipeline_data_new.json'),
        #     test_data('pipeline_data_new.json'),
        #     steps=['fmrds'],
        #     verbose=True,
        #     # max_iterations=5
        # )

        o = res.optimizer
        r, t = o.rs, o.ts
        uuh = fopt.ForceOptimizer(r, t, internals=o.internals)
        print(o.force_coeffs.shape,
              uuh.force_coeffs.shape,
              uuh.to_data().force_coeffs.shape)
        uuh2 = fopt.ForceOptimizer.from_data(uuh.to_data())
        print(uuh.force_coeffs.shape,
              uuh2.force_coeffs.shape)

    @unittest.skip
    def test_Refinements(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        traj = utils.read_namedtuple(test_data('reopt_test.json'))
        yeesh = gen_prods.refine_trajectory(
            traj,
            refine_endpoints=False,
            # num_images=6,
            # profile_generator=None,
            profile_generator='pys-ts',
            # profile_generator='pys-cos',
            # ts_opt_generator='pys-ts',
            # spring_constant=.01,
            # max_iterations=50,
            max_refinement_iterations=50,
            logger=True,
            # coord_type='dlc'
            optimizer_settings=dict(coord_diff_thresh=0.0005)
        )
        trajt.compare_profiles(yeesh, marker='o').show()

    @unittest.skip
    def test_InternalsForces(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        traj = pipeline.OptimizedForceResults.from_file(test_data('pipeline_data.json'))

        import subprocess
        import memray
        import os
        try:
            os.remove(os.path.expanduser("~/Desktop/memprof.out"))
        except:
            ...
        with memray.Tracker("/Users/Mark/Desktop/memprof.out"):
            opt = traj.optimizer.reoptimize_internals_with_force(
                'dihedrals',
                max_internals=2,
                max_iterations=100
            )

        try:
            os.remove(os.path.expanduser("~/Desktop/memprof_graph.html"))
        except:
            ...
        subprocess.run(["memray", "flamegraph", os.path.expanduser("~/Desktop/memprof.out"), "-o", os.path.expanduser("~/Desktop/memprof_graph.html")])
        subprocess.run(["open",  os.path.expanduser("~/Desktop/memprof_graph.html")])

    @unittest.skip
    def test_RandomForces(self):
        import McUtils.Numputils as nput

        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        traj = pipeline.OptimizedForceResults.from_file(test_data('pipeline_data.json'))
        optimizer = traj.optimizer
        dir2 = nput.find_basis(
            np.random.normal(size=(10, 3 * len(optimizer.ts.atoms))).T,
            method='qr'
        ).T
        opt = traj.optimizer.reoptimize_with_force(
            [0, 1],
            displacements=dir2,
            max_iterations=5
        )

    @unittest.skip
    def test_HydrostaticForces(self):

        import McUtils.Numputils as nput
        from McUtils.Data import UnitsData
        from Psience.Molecools import Molecule

        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        traj = pipeline.OptimizedForceResults.from_file(test_data('pipeline_data.json'))
        # optimizer = traj.optimizer
        # dir2 = nput.find_basis(
        #     np.random.normal(size=(10, 3 * len(optimizer.ts.atoms))).T,
        #     method='qr'
        # ).T


        # def apply_hydrostatic(ts:Molecule, coords, surface_points=500, radius_scaling=1.2):
        #     coords = np.asanyarray(coords).reshape((-1, 3))
        #     surf = ts.modify(coords=coords).get_surface(samples=surface_points, radius_scaling=radius_scaling).get_triangulation()
        #     groups, _ = nput.group_by(np.arange(len(surf.tri_map)), surf.tri_map)
        #     areas = surf.surface_area(return_components=True)
        #     area_fractions = areas / np.sum(areas)
        #     scaled_normals = area_fractions[..., np.newaxis] * surf.normals
        #     force = np.zeros_like(coords)
        #     for atom, inds in zip(*groups):
        #         force[atom] = np.sum(scaled_normals[inds,], axis=0)
        #     return force.flatten()[np.newaxis]
        #
        #     # surf
        #     # compute terms normal to the surface
        #     ...

        # fig = traj.optimizer.rs.plot(backend='x3d')
        # surf = traj.optimizer.rs.get_surface(samples=500, radius_scaling=1.2).get_triangulation()
        # surf.plot(figure=fig)
        # fig.show()

        _, _, opt = traj.optimizer.reoptimize_with_pressure(5,
                                                            "Gigapascals",
                                                            pressure_model='cylinder',
                                                            pressure_options={
                                                                'axis':'-c',
                                                                'radius':1 * UnitsData.convert("Angstroms", "BohrRadius"),
                                                                'bidirectional':True
                                                            },
                                                            max_iterations=50,
                                                            logger=True)

        print(
            (opt.transition_state_energy - opt.reactant_energy) * UnitsData.convert("Hartrees", "Kilocalories/Mole"),
            (opt.force_modified_transition_state_energy - opt.force_modified_reactant_energy)* UnitsData.convert("Hartrees", "Kilocalories/Mole")
        )

        traj.plot_profile(fmrd=opt, force_modified='both').show()

        traj.optimizer.rs.plot([
            opt.reactant_geom,
            opt.force_modified_reactant_geom
        ]).show()

    @unittest.skip
    def test_ExportNew(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        res = pipeline.OptimizedForceResults.from_file(test_data('pipeline_data.json'))

        npdat = utils.dumps_namedtuple(res.to_data(), mode='npz')
        print(utils.loads_namedtuple(npdat, mode='npz', decompress=True))

        res.compare_profiles().show()

    @unittest.skip
    def test_LoadPipelineNPZ(self):
        uuh = pipeline.read_compressed_pipeline_data('/Users/Mark/Documents/Postdoc/Projects/CMCC/disub_alt.npz')

    @unittest.skip
    def test_PressureRigidScan(self):
        import McUtils.Numputils as nput
        from McUtils.Data import UnitsData
        from Psience.Molecools import Molecule

        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        traj = pipeline.OptimizedForceResults.from_file(test_data('pipeline_data.json'))
        _, x_r, x_t = traj.optimizer.get_pressure_distorted_geometries(steps=5,
                                                                       pressure_model='hcff',
                                                                       # pressure_options={
                                                                       #     'axis': '-b',
                                                                       #     'radius': 10 * UnitsData.convert("Angstroms",
                                                                       #                                     "BohrRadius"),
                                                                       #     'bidirectional': True
                                                                       # }
                                                                       )
        # traj.reactant.plot(x_r).show()
        traj.transition_state.plot(x_t, principle_axes=True).show()

    @unittest.skip
    def test_PressureFromMols(self):
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        from cmcc_reactions.optimal_directions import ForceOptimizer
        from cmcc_reactions.reaction_data_analysis import ForceModifiedReactionAnalyzer
        from Psience.Molecools import Molecule
        from McUtils.Data import UnitsData

        ts = Molecule.from_string(
            '''23

C	-2.25046  -0.72777   0.38674
C	-2.84525   0.27119  -0.40824
C	-2.05528   1.43188  -0.34975
C	-1.00007   1.21623   0.53068
H	-3.68826   0.12095  -1.07529
H	-2.18971   2.31405  -0.96724
C	-1.31163  -0.01193   1.33697
H	-1.89312   0.2926    2.22416
H	-0.44662  -0.58299   1.67761
H	 3.36458   0.84622   1.183
C	 3.53875   0.60349   0.1304
H	 4.13059   1.38582  -0.34751
O	 2.30512   0.55523  -0.59458
O	 1.64912  -1.11199   0.79362
C	 1.41406  -0.37994  -0.15824
C	 0.19176  -0.37027  -0.96198
C	-0.73573  -1.41088  -0.8544
H	 4.06087  -0.35627   0.07649
H	 0.17385   0.31187  -1.80252
H	-0.25144   1.95016   0.8072
H	-0.46167  -2.25662  -0.22932
H	-1.34765  -1.65825  -1.71479
H	-2.76226  -1.64549   0.66279''',
            units='Angstroms',
            energy_evaluator='aimnet2'
        )
        rs = Molecule.from_string(
            '''23

C	-2.596883   -0.465922    0.826097
C	-3.007938    0.362995   -0.157348
C	-2.176283    1.573125   -0.150206
C	-1.262554    1.479210    0.838924
H	-3.822674    0.183416   -0.851937
H	-2.294223    2.402121   -0.840972
C	-1.440859    0.166969    1.555182
H	-1.657069    0.307193    2.625854
H	-0.530419   -0.449785    1.507801
H	 3.184941    0.970467    1.093409
C	 3.674170    0.595730    0.190227
H	 4.313221    1.363308   -0.247407
O	 2.702223    0.281677   -0.817778
O	 1.754811   -1.178029    0.628884
C	 1.777653   -0.640983   -0.463418
C	 0.829399   -0.888538   -1.575622
C	-0.113274   -1.828218   -1.473814
H	 4.260314   -0.290616    0.448467
H	 0.945444   -0.275067   -2.463748
H	-0.507370    2.212516    1.098686
H	-0.199381   -2.428540   -0.573233
H	-0.817422   -2.019587   -2.277227
H	-3.015828   -1.433443    1.079180''',
            units='Angstroms',
            energy_evaluator='aimnet2'
        )

        prod = Molecule.from_string('''
23

C	-2.130472   -0.872239    0.012830
C	-2.837943    0.423768   -0.358236
C	-2.036169    1.437416   -0.003026
C	-0.785002    0.832289    0.615247
H	-3.768536    0.486030   -0.914065
H	-2.171380    2.494087   -0.211641
C	-1.380485   -0.424799    1.289712
H	-2.054779   -0.174685    2.114386
H	-0.623578   -1.141357    1.624103
H	 3.632781    0.252404    1.177904
C	 3.620668    0.446425    0.101839
H	 4.203134    1.337891   -0.133497
O	 2.285694    0.727776   -0.347439
O	 1.678608   -1.328818    0.357919
C	 1.385915   -0.265794   -0.151448
C	 0.003633    0.166438   -0.596488
C	-0.924072   -1.025404   -0.977450
H	 4.025934   -0.425731   -0.418228
H	 0.120871    0.893828   -1.402633
H	-0.171319    1.499026    1.225121
H	-0.408315   -1.974347   -0.806154
H	-1.237369   -0.984186   -2.024394
H	-2.758306   -1.763227    0.087370''',
            units='Angstroms',
            energy_evaluator='aimnet2'
        )

        # prod = prod.optimize(mode='pysis', method='rfo', max_iterations=200, logger=True)
        # prod_data = gen_prods.InitialProductData(
        #     atoms=prod.atoms,
        #     coords=prod.coords,
        #     smiles=None,
        #     bonds=None,
        #     energy=prod.calculate_energy(),
        #     breakpoints=None,
        #     evaluator='aimnet2',
        #     optimization_settings=None
        # )
        # utils.write_namedtuple('/Users/Mark/Desktop/methacrylate_prod.npz', prod_data)
        #
        # return
        # rs.plot(display_atom_numbers=True).show()
        # ts.plot(highlight_atoms=[0, 3, 15, 16]).show()

        rs = rs.optimize(mode='pysis', method='rfo', max_iterations=200, logger=True)
        ts = ts.optimize(mode='pysis', method='ts', max_iterations=100, logger=True)

        fopt = ForceOptimizer(rs, ts, precompute_modes=False)
        # fopt._debug_show_force_vectors = True
        # print(fopt.ts.calculate_energy() - fopt.rs.calculate_energy())
        # _, x_r, x_t = fopt.get_pressure_distorted_geometries(steps=5,
        #                                                      pressure_model='xhcff',
        #                                                      # pressure_options={
        #                                                      #     'axis': '-b',
        #                                                      #     'radius': 10 * UnitsData.convert("Angstroms",
        #                                                      #                                     "BohrRadius"),
        #                                                      #     'bidirectional': True
        #                                                       # }
        #                                                      )

        # de_data, (x_r, x_t) = fopt.get_pressure_distortion_energies(
        #     steps=15,
        #     return_geometries=True,
        #     # pressure_model='xhcff',
        #     pressure_model='cylinder',
        #     pressure_options={
        #         'axis': '-b',
        #         'radius': 10 * UnitsData.convert("Angstroms", "BohrRadius"),
        #         'bidirectional': True
        #     }
        # )
        # fopt.plot_eng_comp(*de_data).show()
        # fopt.ts.plot(x_t, principle_axes=True).show()

        import McUtils.Numputils as nput
        fmd_r, fmd_t, data = fopt.reoptimize_with_pressure(
            10,
            pressure_units="Gigapascals",
            pressure_model='cavity',
            # optimizer_mode='scipy',
            # optimizer_method='bfgs',
            # pressure_model='cylinder',
            # pressure_model='cylinder',
            # pressure_options={
            #     'axis': lambda coords, _:nput.vec_normalize(
            #         np.average(coords[(0, 3), :], axis=0)
            #         - np.average(coords[(15, 16), :], axis=0)
            #     ),
            #     'centroid': lambda coords:np.average(coords[(0, 3, 15, 16), :], axis=0),
            #     'radius': 4 * UnitsData.convert("Angstroms", "BohrRadius"),
            #     'bidirectional': True
            # },
            max_displacement=.1,
            max_iterations=20,
            logger=True,
        )

        fmd_e_t, fmd_e_r = data.force_modified_transition_state_energy, data.force_modified_reactant_energy
        e_t, e_r = data.transition_state_energy, data.reactant_energy


        print("Baseline: {de} kcal mol^-1".format(
            de=(e_t - e_r) * UnitsData.convert("Hartrees", "Kilocalories/Mole")
        ))
        print("Pressure: {de} kcal mol^-1".format(
            de=(fmd_e_t - fmd_e_r) * UnitsData.convert("Hartrees", "Kilocalories/Mole")
        ))

        fmra = ForceModifiedReactionAnalyzer.from_data(data)
        fmra.animate_reactant_distortion(atom_radius_scaling=1).show()
        fmra.animate_ts_distortion(atom_radius_scaling=1).show()

        # print("Reactants:")
        # print(rs.to_string('xyz', units='Angstroms'))
        # print("Transition States:")
        # print(ts.to_string('xyz', units='Angstroms'))
        # print("Force Modified Reactant:")
        # print(fmd_r.to_string('xyz', units='Angstroms'))
        # print("Force Modified Transition State:")
        # print(fmd_t.to_string('xyz', units='Angstroms'))

        # utils.write_namedtuple('/Users/Mark/Desktop/methacrylate_fmrd_hydrostatic.npz', data)
        fmra.plot_lines().show()


    @unittest.skip
    def test_RigidForceOpts(self):

        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        from cmcc_reactions.optimal_directions import ForceOptimizer
        from cmcc_reactions.reaction_data_analysis import ForceModifiedReactionAnalyzer
        from Psience.Molecools import Molecule
        from McUtils.Data import UnitsData

        ts = Molecule.from_string(
            '''23

C	-2.25046  -0.72777   0.38674
C	-2.84525   0.27119  -0.40824
C	-2.05528   1.43188  -0.34975
C	-1.00007   1.21623   0.53068
H	-3.68826   0.12095  -1.07529
H	-2.18971   2.31405  -0.96724
C	-1.31163  -0.01193   1.33697
H	-1.89312   0.2926    2.22416
H	-0.44662  -0.58299   1.67761
H	 3.36458   0.84622   1.183
C	 3.53875   0.60349   0.1304
H	 4.13059   1.38582  -0.34751
O	 2.30512   0.55523  -0.59458
O	 1.64912  -1.11199   0.79362
C	 1.41406  -0.37994  -0.15824
C	 0.19176  -0.37027  -0.96198
C	-0.73573  -1.41088  -0.8544
H	 4.06087  -0.35627   0.07649
H	 0.17385   0.31187  -1.80252
H	-0.25144   1.95016   0.8072
H	-0.46167  -2.25662  -0.22932
H	-1.34765  -1.65825  -1.71479
H	-2.76226  -1.64549   0.66279''',
            units='Angstroms',
            energy_evaluator='aimnet2'
        )
        rs = Molecule.from_string(
            '''23

C	-2.596883   -0.465922    0.826097
C	-3.007938    0.362995   -0.157348
C	-2.176283    1.573125   -0.150206
C	-1.262554    1.479210    0.838924
H	-3.822674    0.183416   -0.851937
H	-2.294223    2.402121   -0.840972
C	-1.440859    0.166969    1.555182
H	-1.657069    0.307193    2.625854
H	-0.530419   -0.449785    1.507801
H	 3.184941    0.970467    1.093409
C	 3.674170    0.595730    0.190227
H	 4.313221    1.363308   -0.247407
O	 2.702223    0.281677   -0.817778
O	 1.754811   -1.178029    0.628884
C	 1.777653   -0.640983   -0.463418
C	 0.829399   -0.888538   -1.575622
C	-0.113274   -1.828218   -1.473814
H	 4.260314   -0.290616    0.448467
H	 0.945444   -0.275067   -2.463748
H	-0.507370    2.212516    1.098686
H	-0.199381   -2.428540   -0.573233
H	-0.817422   -2.019587   -2.277227
H	-3.015828   -1.433443    1.079180''',
            units='Angstroms',
            energy_evaluator='aimnet2'
        )

        prod = Molecule.from_string('''23

    C	-2.130472   -0.872239    0.012830
    C	-2.837943    0.423768   -0.358236
    C	-2.036169    1.437416   -0.003026
    C	-0.785002    0.832289    0.615247
    H	-3.768536    0.486030   -0.914065
    H	-2.171380    2.494087   -0.211641
    C	-1.380485   -0.424799    1.289712
    H	-2.054779   -0.174685    2.114386
    H	-0.623578   -1.141357    1.624103
    H	 3.632781    0.252404    1.177904
    C	 3.620668    0.446425    0.101839
    H	 4.203134    1.337891   -0.133497
    O	 2.285694    0.727776   -0.347439
    O	 1.678608   -1.328818    0.357919
    C	 1.385915   -0.265794   -0.151448
    C	 0.003633    0.166438   -0.596488
    C	-0.924072   -1.025404   -0.977450
    H	 4.025934   -0.425731   -0.418228
    H	 0.120871    0.893828   -1.402633
    H	-0.171319    1.499026    1.225121
    H	-0.408315   -1.974347   -0.806154
    H	-1.237369   -0.984186   -2.024394
    H	-2.758306   -1.763227    0.087370''',
                                    units='Angstroms',
                                    energy_evaluator='aimnet2'
                                    )

        # prod = prod.optimize(mode='pysis', method='rfo', max_iterations=200, logger=True)
        # prod_data = gen_prods.InitialProductData(
        #     atoms=prod.atoms,
        #     coords=prod.coords,
        #     smiles=None,
        #     bonds=None,
        #     energy=prod.calculate_energy(),
        #     breakpoints=None,
        #     evaluator='aimnet2',
        #     optimization_settings=None
        # )
        # utils.write_namedtuple('/Users/Mark/Desktop/methacrylate_prod.npz', prod_data)
        #
        # return
        # rs.plot(display_atom_numbers=True).show()
        # ts.plot(highlight_atoms=[0, 3, 15, 16]).show()

        rs = rs.optimize(mode='pysis', method='rfo', max_iterations=200, logger=True)
        ts = ts.optimize(mode='pysis', method='ts', max_iterations=100, logger=True)

        internals = rs.get_bond_zmatrix()
        fopt = ForceOptimizer(rs, ts,
                              internals=internals,
                              fragment_indices=1,#np.setdiff1d(rs.fragment_indices[1], (0, 16, 3, 15)),
                              precompute_modes=False)
        # fopt._debug_show_force_vectors = True

        # which = fopt.get_selected_internals('dihedrals', max_internals=5, max_internals_ranks=[-1])
        # fopt.internal_mols[0].animate_coordinate(which[0], backend='x3d').show()
        # fopt.internal_mols[1].animate_coordinate(which[0], backend='x3d').show()
        # return

        fmrd_res = fopt.reoptimize_internals_with_force('dihedrals',
                                                        magnitude=1000,
                                                        max_internals=1,
                                                        max_internals_ranks=[-1],
                                                        rigid=True,
                                                        return_selected=True)
        fmrd_res, (which2, gammas) = fmrd_res
        # fopt.internal_mols[0].animate_coordinate(which2[0], backend='x3d').show()
        # fopt.internal_mols[1].animate_coordinate(which2[0], backend='x3d').show()

        fmd_r, fmd_t, data = fmrd_res[0]

        fmd_e_t, fmd_e_r = data.force_modified_transition_state_energy, data.force_modified_reactant_energy
        e_t, e_r = data.transition_state_energy, data.reactant_energy

        print("Baseline: {de} kcal mol^-1".format(
            de=(e_t - e_r) * UnitsData.convert("Hartrees", "Kilocalories/Mole")
        ))
        print("Distortion: {de} kcal mol^-1".format(
            de=(fmd_e_t - fmd_e_r) * UnitsData.convert("Hartrees", "Kilocalories/Mole")
        ))

        fmra = ForceModifiedReactionAnalyzer.from_data(data)
        fmra.animate_reactant_distortion().show()
        fmra.animate_ts_distortion().show()
        fmra.plot_lines(bonds=[(0, 16), (3, 15)]).show()


    def test_Sterics(self):

        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        from cmcc_reactions.optimal_directions import ForceOptimizer
        from cmcc_reactions.reaction_data_analysis import ForceModifiedReactionAnalyzer
        from Psience.Molecools import Molecule
        from McUtils.Data import UnitsData

        ts = Molecule.from_string(
            '''23

C	-2.25046  -0.72777   0.38674
C	-2.84525   0.27119  -0.40824
C	-2.05528   1.43188  -0.34975
C	-1.00007   1.21623   0.53068
H	-3.68826   0.12095  -1.07529
H	-2.18971   2.31405  -0.96724
C	-1.31163  -0.01193   1.33697
H	-1.89312   0.2926    2.22416
H	-0.44662  -0.58299   1.67761
H	 3.36458   0.84622   1.183
C	 3.53875   0.60349   0.1304
H	 4.13059   1.38582  -0.34751
O	 2.30512   0.55523  -0.59458
O	 1.64912  -1.11199   0.79362
C	 1.41406  -0.37994  -0.15824
C	 0.19176  -0.37027  -0.96198
C	-0.73573  -1.41088  -0.8544
H	 4.06087  -0.35627   0.07649
H	 0.17385   0.31187  -1.80252
H	-0.25144   1.95016   0.8072
H	-0.46167  -2.25662  -0.22932
H	-1.34765  -1.65825  -1.71479
H	-2.76226  -1.64549   0.66279''',
            units='Angstroms',
            energy_evaluator='aimnet2'
        )
        rs = Molecule.from_string(
            '''23

C	-2.596883   -0.465922    0.826097
C	-3.007938    0.362995   -0.157348
C	-2.176283    1.573125   -0.150206
C	-1.262554    1.479210    0.838924
H	-3.822674    0.183416   -0.851937
H	-2.294223    2.402121   -0.840972
C	-1.440859    0.166969    1.555182
H	-1.657069    0.307193    2.625854
H	-0.530419   -0.449785    1.507801
H	 3.184941    0.970467    1.093409
C	 3.674170    0.595730    0.190227
H	 4.313221    1.363308   -0.247407
O	 2.702223    0.281677   -0.817778
O	 1.754811   -1.178029    0.628884
C	 1.777653   -0.640983   -0.463418
C	 0.829399   -0.888538   -1.575622
C	-0.113274   -1.828218   -1.473814
H	 4.260314   -0.290616    0.448467
H	 0.945444   -0.275067   -2.463748
H	-0.507370    2.212516    1.098686
H	-0.199381   -2.428540   -0.573233
H	-0.817422   -2.019587   -2.277227
H	-3.015828   -1.433443    1.079180''',
            units='Angstroms',
            energy_evaluator='aimnet2'
        )

        prod = Molecule.from_string('''23

            C	-2.130472   -0.872239    0.012830
            C	-2.837943    0.423768   -0.358236
            C	-2.036169    1.437416   -0.003026
            C	-0.785002    0.832289    0.615247
            H	-3.768536    0.486030   -0.914065
            H	-2.171380    2.494087   -0.211641
            C	-1.380485   -0.424799    1.289712
            H	-2.054779   -0.174685    2.114386
            H	-0.623578   -1.141357    1.624103
            H	 3.632781    0.252404    1.177904
            C	 3.620668    0.446425    0.101839
            H	 4.203134    1.337891   -0.133497
            O	 2.285694    0.727776   -0.347439
            O	 1.678608   -1.328818    0.357919
            C	 1.385915   -0.265794   -0.151448
            C	 0.003633    0.166438   -0.596488
            C	-0.924072   -1.025404   -0.977450
            H	 4.025934   -0.425731   -0.418228
            H	 0.120871    0.893828   -1.402633
            H	-0.171319    1.499026    1.225121
            H	-0.408315   -1.974347   -0.806154
            H	-1.237369   -0.984186   -2.024394
            H	-2.758306   -1.763227    0.087370''',
                                    units='Angstroms',
                                    energy_evaluator='aimnet2'
                                    )


        # prod = prod.optimize(mode='pysis', method='rfo', max_iterations=200, logger=True)
        # prod_data = gen_prods.InitialProductData(
        #     atoms=prod.atoms,
        #     coords=prod.coords,
        #     smiles=None,
        #     bonds=None,
        #     energy=prod.calculate_energy(),
        #     breakpoints=None,
        #     evaluator='aimnet2',
        #     optimization_settings=None
        # )
        # utils.write_namedtuple('/Users/Mark/Desktop/methacrylate_prod.npz', prod_data)
        #
        # return
        # rs.plot(display_atom_numbers=True).show()
        # ts.plot(highlight_atoms=[0, 3, 15, 16]).show()

        # rs = rs.optimize(mode='pysis', method='rfo', max_iterations=200)
        rs.modify(coords=[[-5.17852785, -0.5702728, 1.88441103],
                          [-5.54956468, 0.71849346, -0.2612763],
                          [-3.41217425, 2.44967796, -0.65371732],
                          [-1.75827867, 2.18121372, 1.2402138],
                          [-7.12634655, 0.52129859, -1.54238332],
                          [-3.21716378, 3.6987746, -2.2546904],
                          [-2.76157906, 0.278203, 3.08535031],
                          [-3.09024025, 1.1128944, 4.94702848],
                          [-1.44325623, -1.28453075, 3.37485686],
                          [5.60599726, 1.86448078, 2.28898434],
                          [6.58723769, 1.38085643, 0.54733104],
                          [7.66704781, 2.98256345, -0.13327283],
                          [4.81254645, 0.7980625, -1.40377519],
                          [3.37442623, -2.38242883, 1.01321266],
                          [3.27302274, -1.16958319, -0.91036453],
                          [1.44564317, -1.58108074, -2.99327646],
                          [-0.43341279, -3.20964043, -2.72572653],
                          [7.81949631, -0.22504678, 0.89926876],
                          [1.65122534, -0.4470092, -4.67593706],
                          [0.06109034, 3.09154467, 1.42019802],
                          [-0.62957867, -4.28276737, -1.00259567],
                          [-1.83502836, -3.45704542, -4.18710659],
                          [-6.38909548, -2.00507506, 2.67724378]])
        # ts = ts.optimize(mode='pysis', method='ts', max_iterations=100)
        ts.modify(coords=[[-4.23705432, -1.28118781, 0.7729455],
                          [-5.15557595, 0.63656345, -0.80710441],
                          [-3.49571837, 2.69101028, -0.71292047],
                          [-1.57424739, 2.1469359, 0.9659398],
                          [-6.70869671, 0.42514645, -2.11988187],
                          [-3.52244096, 4.28272382, -1.99019307],
                          [-2.36994421, -0.05734278, 2.53149949],
                          [-3.39636713, 0.62278688, 4.19032199],
                          [-0.85797006, -1.29830522, 3.16420706],
                          [5.88639691, 1.68459917, 2.38192],
                          [6.33171981, 1.39937868, 0.39375325],
                          [7.31961015, 3.02852809, -0.36245937],
                          [4.07897804, 1.16920227, -1.0648456],
                          [3.10557335, -2.20565703, 1.33176506],
                          [2.55995036, -0.77733233, -0.37393262],
                          [0.24684349, -0.84549836, -1.85578447],
                          [-1.46639409, -2.7800723, -1.47819944],
                          [7.48730166, -0.29322758, 0.21371082],
                          [0.17742473, 0.35813596, -3.5035662],
                          [-0.02313225, 3.37780772, 1.43591499],
                          [-0.94401441, -4.23142231, -0.14070224],
                          [-2.67619796, -3.37972009, -3.01072528],
                          [-5.29351647, -2.94980018, 1.29627762]])

        internals = rs.get_bond_zmatrix()
        fopt = ForceOptimizer(rs, ts,
                              internals=internals,
                              fragment_indices=1,  # np.setdiff1d(rs.fragment_indices[1], (0, 16, 3, 15)),
                              precompute_modes=False)

        (s_r, s_t), (v, x_r, x_t) = fopt.get_distortion_steric_repulsions(
            'dihedrals',
            max_internals=1,
            generate_displacement_function=fopt.generate_internal_distortions,
            disp_min=-1,
            disp_max=0,
            density=2,
            return_breakdowns=True
        )
        pts_r = [np.concatenate(p) for p in s_r[0]]
        vals_r = [np.concatenate(v) for v in s_r[1]]
        pts_t = [np.concatenate(p) for p in s_t[0]]
        vals_t = [np.concatenate(v) for v in s_t[1]]


        # vals_t = np.concatenate(s_t[1])
        # print(s_r)
        # print(s_t)
        # print(np.diff(pts_r, axis=0))

        # fopt.rs.plot(x_r,
        #              # transparency=.5,
        #              # bonds=False,
        #              # atom_radius_scaling=1,
        #              annotation_function=lambda mol, i, geom: [
        #                  mplt.Sphere(p * UnitsData.bohr_to_angstroms,
        #                              .1, color='black')
        #                  for p in pts_r[i]
        #              ]
        #              ).show()
        import McUtils.Plots as mplt
        import McUtils.Numputils as nput

        keep_pos_r = np.any(
            np.abs(
                nput.vec_rescale(
                    np.moveaxis(np.array(vals_r), 0, -1),
                    [-1, 1],
                    [np.min(vals_r), np.max(vals_r)]
                )
            ) > .95,
            axis=-1
        )
        keep_pos_t = np.any(
            np.abs(
                nput.vec_rescale(
                    np.moveaxis(np.array(vals_t), 0, -1),
                    [-1, 1],
                    [np.min(vals_t), np.max(vals_t)]
                )
            ) > .5,
            axis=-1
        )
        pts_r = [p[keep_pos_r] for p in pts_r]
        vals_r = [v[keep_pos_r] for v in vals_r]
        pts_t = [p[keep_pos_t] for p in pts_t]
        vals_t = [v[keep_pos_t] for v in vals_t]

        min_max = np.min(np.concatenate([
            np.concatenate(vals_r),
            np.concatenate(vals_t)
        ])), np.max(np.concatenate([
                    np.concatenate(vals_r),
                    np.concatenate(vals_t)
                ]))


        colors_r = [
            mplt.prep_color(palette='coolwarm', blending=nput.vec_rescale(
                v,
                [0, 1],
                min_max
            )**2)
            for v in vals_r
        ]

        colors_t = [
            mplt.prep_color(palette='coolwarm', blending=nput.vec_rescale(
                v,
                [0, 1],
                min_max
            ) ** 2)
            for v in vals_t
        ]

        import McUtils.Jupyter as interactive
        uuh = interactive.Grid([
            [fopt.rs.plot(x_r,
                          annotation_function=lambda mol, i, geom: [
                              mplt.Sphere(p * UnitsData.bohr_to_angstroms, .2, color=c)
                              for p,c in zip(pts_r[i], colors_r[i])
                          ]
                          ).to_widget(),
             fopt.ts.plot(x_t,
                          annotation_function=lambda mol, i, geom: [
                              mplt.Sphere(p * UnitsData.bohr_to_angstroms, .2, color=c)
                              for p, c in zip(pts_t[i], colors_t[i])
                          ]
                          ).to_widget()
             ]
        ], dynamic=False).to_widget().display()

        return


if __name__ == '__main__':
    os.chdir(root)
    unittest.main('tests.CMCCTests')