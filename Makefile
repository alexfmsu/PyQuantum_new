tree:
	tree -I "__pycache__|*.pyc|out|*.csv|*.gz|__init__.py|stuff|*.json|build|dist|LICENSE|README.md|Makefile|logs|PyQuantum.egg-info|sink_single_out|setup.py|install_python*" > edit_tree

clean:
	rm *.out *.err
