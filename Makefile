.PHONY: all clean deepclean style lint

all: penalty_writeup.pdf esrd_analytic.html appendix.pdf

penalty_writeup.pdf: penalty_writeup.qmd esrd_analytic.qmd fairpenalties.bib code/simpop.py code/pen_regression.py code/comps.py tables/logistic_baseline.csv tables/tab1_baseline.csv tables/penalty_res_*.csv tables/penalty_res_al0.csv tables/comp_res_*.csv tables/esrd_bigtab.csv tables/esrd_comp.csv _extensions/quarto-journals/jasa/_extension.yml _extensions/quarto-journals/jasa/american-statistical-association.csl _extensions/quarto-journals/jasa/asa.bst _extensions/quarto-journals/jasa/partials/_include-in-header.tex _extensions/quarto-journals/jasa/partials/before-body.tex
	uv run quarto render penalty_writeup.qmd --to html,jasa-pdf,docx

esrd_analytic.html: esrd_analytic.qmd code/pen_regression.py
	uv run quarto render esrd_analytic.qmd

appendix.pdf: appendix.qmd code/pen_regression.py code/simpop.py tables/tab1_baseline.csv tables/penalty_res_*.csv tables/penalty_res_al0.csv tables/drawcounts_*.csv tables/comp_res_*.csv
	uv run quarto render appendix.qmd

# note: clean 
clean:
	rm -f figures/simres*.png esrd_analytic.html penalty_writeup.html

deepclean: 
	rm -f figures/*.png figures/*.pdf

style:
	uvx ruff format .

lint:
	uvx ruff check .
