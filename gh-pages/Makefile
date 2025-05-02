.PHONY: all clean distClean build gen

all: clean build gen

clean:
	$(RM) -r public content

distclean: clean
	$(RM) -r static

build:
	./scripts/build_all.sh $(wildcard ../data/*yaml)

gen:
	./scripts/gen_bundles.sh
