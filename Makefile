.PHONY: quality style test docs utils

check_dirs := .

# Check that source code meets quality standards

# this target runs checks on all files
quality:
	ruff check $(check_dirs)
	ruff format --check $(check_dirs)
# Format source code automatically and check is there are any problems left that need manual fixing
style:
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)

clean:
	$(MAKE) style
	$(MAKE) quality