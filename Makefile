.PHONY: test synth
test:
	pytest -q
synth:
	cd infra && cdk synth -c config=../configs/smoke.yaml --quiet
