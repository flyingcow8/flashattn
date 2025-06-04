
clean_dist:
	rm -rf dist/*

create_dist: clean_dist
	python setup.py sdist

upload_package: create_dist
	twine upload dist/*

cplus_api:
	mkdir -p build_cpp
	cd build_cpp; cmake \
	-DMACA_PATH=${MACA_PATH} \
	-DCPACK_GENERATOR=DEB \
	-DCMAKE_INSTALL_PREFIX=./install \
	.. && make -j64 && make install

clean_capi:
	rm -rf build_cpp

clean:
	rm -rf build*
