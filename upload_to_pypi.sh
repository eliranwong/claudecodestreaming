python3 -m pip install build twine
rm -rf dist build
python3 -m build
twine upload dist/*