# Build docs


## Step 1
```
pip install mkdocs mkdocs-material mkdocstrings mkdocstrings-python
```

## Step 2
Generate API docs automatically:

```
python docs/generate_api_docs.py
```

## Step 3

Build site:

```
mkdocs build
```


## Alternatively, serve it for developing purpose.

Serve server:

```
mkdocs serve
```