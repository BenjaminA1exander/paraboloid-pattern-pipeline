# Earlier implementation (not used by the pipeline)

The cutting-pattern optimiser as first written for the NPL airship envelope, from which the
`vares_*` modules were consolidated: `optimise_cutting_pattern_npl_gore_chen.py` (dense Newton
driver, solver constants now in `vares_settings.py`), `optimise_cutting_pattern_paraboloid_gore_chen.py`
(pin rule and residual field, now `vares_setup.py`), `flatten_semicylinder_validation.py` (the
least-squares flattening, now `vares_flatten.py`), `bilinear_quad_lib.py` and
`single_bilinear_quad_covariant.py` (quad shape functions and metrics, now in `element_types.py`),
`npl_profile.py` (the airship profile). Kept for reference; `MeshStudy_paraboloid_vares.run_one`
still uses the dense loop of the first file.
