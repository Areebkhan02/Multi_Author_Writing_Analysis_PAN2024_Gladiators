# Manual steps for submission to tira

Build the docker image via:

```
docker build -t pan24-gladiators .
```

Test it locally:

```
tira-run \
	--input-dataset multi-author-writing-style-analysis-2024/multi-author-spot-check-20240428-training \
	--image pan24-gladiators
```

