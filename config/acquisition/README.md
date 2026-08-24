# Acquisition Protocol YAML

Aramina acquisition records are research-prototype inputs for XRD and
attenuation measurements. They support radiologist-led clinical decision
support research and are not autonomous diagnosis records. The 6/9/12 point
variants count total patient points, not points per breast; recorded `alpha`
controls symmetric width-proportional placement and remains pending hardware
validation.

- [Protocol schema](schema/aramina_acquisition_protocol_v0_1.yaml)
- [Validated example](../../examples/acquisition/aramina_acquisition_protocol_v0_1.yaml)
- [Human-readable contract](../../docs/contracts/aramina_acquisition_protocol_v0_1.md)

Validate a record with:

```bash
python -c "from aramina.acquisition_contract import load_acquisition_protocol; load_acquisition_protocol('examples/acquisition/aramina_acquisition_protocol_v0_1.yaml')"
```
