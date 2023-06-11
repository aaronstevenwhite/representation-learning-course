# Representation Learning for Syntactic and Semantic Theory

This repo contains materials for [a course](https://blogs.umass.edu/lingstitute2023/courses/representation-learning-for-syntactic-and-semantic-theory/) on *Representation Learning for Syntactic and Semantic Theory* given by [Aaron Steven White](http://aaronstevenwhite.io/) at the [2023 Linguistic Society of America Institute](https://blogs.umass.edu/lingstitute2023), held at the [University of Massachusetts, Amherst](https://www.umass.edu/) from June 19–July 14, 2023. 

## About the course

Experimental methods and corpus annotation are becoming increasingly important tools in the development of syntactic and semantic theories. And while regression-based approaches to the analysis of experimental and corpus data are widely known, methods for inducing expressive syntactic and semantic representations from such data remain relatively underused. Such methods have only recently become feasible due to advances in machine learning and the availability of large-scale datasets of acceptability and inference judgments; and they hold promise because they allow theoreticians (i) to design analyses directly in terms of the theoretical constructs of interest and (ii) to synthesize multiple sources and types of data within a single model.

The broad area of machine learning that techniques for syntactic and semantic representation induction come from is known as representation learning; and while such techniques are now common in the natural language processing (NLP) literature, their use is largely confined either to models focused on particular NLP tasks, such as question answering or information extraction, or to ‘probing’ the representations of existing NLP models. As such, it remains difficult to see this literature’s relevance for theoreticians. This course aims to demonstrate that relevance by focusing on the use of representation learning for developing syntactic and semantic theories.

## About the instructor

[Aaron Steven White](http://aaronstevenwhite.io/) is an Associate Professor of [Linguistics](http://www.sas.rochester.edu/lin/) and [Computer Science](https://www.cs.rochester.edu/) at the [University of Rochester](https://rochester.edu/), where he directs the [Formal and Computational Semantics lab](http://factslab.io/) (FACTS.lab). [His research](http://aaronstevenwhite.io/research) investigates the relationship between linguistic expressions and conceptual categories that undergird the human ability to convey information about possible past, present, and future configurations of things in the world. 

In addition to being a principal investigator on numerous federally funded grants and contracts, White is the recipient of [a National Science Foundation Faculty Early Career Development (CAREER) award](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2237175). His work has appeared in a variety linguistics, cognitive science, and natural language processing venues, including Semantics & Pragmatics, Glossa, Language Acquisition, Cognitive Science, Cognitive Psychology, Transactions of the Association for Computational Linguistics, and Empirical Methods in Natural Language Processing.

## About the site

The site itself is built using [Quarto](https://quarto.org/). The source files for this site are available on github at [`aaronstevenwhite/representation-learning-course`](https://github.com/aaronstevenwhite/representation-learning-course).

## Installation

The site itself is built using [Quarto](https://quarto.org/). The source files for this site are available on github at [`aaronstevenwhite/representation-learning-course`](https://github.com/aaronstevenwhite/representation-learning-course). You can obtain the files by cloning this repo.

```bash
git clone https://github.com/aaronstevenwhite/representation-learning-course.git
```

All further code on this page assumes that you are inside of this cloned repo.

```bash
cd representation-learning-course
```

### Installing Quarto and extensions

To build this site, you will need to [install Quarto](https://quarto.org/docs/get-started/) as well as its [`include-code-files` extension](https://github.com/quarto-ext/include-code-files).

```bash
quarto add quarto-ext/include-code-files
```

This extension is mainly used for including external [STAN](https://mc-stan.org/) files.

### Building the Docker container

All pages that have executed code blocks are generated from jupyter notebooks, which were run within a Docker container constructed using the Dockerfile contained in this repo. 

```dockerfile
FROM jupyter/datascience-notebook:notebook-6.5.4

RUN pip install --upgrade pip cmdstanpy==1.1.0 arviz==0.15.1 torch==2.0.1 'transformers[torch]' &&\
    python -c "from cmdstanpy import install_cmdstan; install_cmdstan(version='2.32.2')"
```

The image can be built using:

```bash
docker build -t representation-learning-course .
```

A container based on this image can then be constructed using:

```bash
docker run -it --rm -p 8888:8888 -v "${PWD}":/home/jovyan/work representation-learning-course
```

To access jupyter, simply copy the link provided when running this command. You can change the port that docker forwards to by changing the first `8888` in the `-p 8888:8888` option. Just remember to correspondingly change the port you attempt to access in your browser.

## License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" href="http://purl.org/dc/dcmitype/Text" property="dct:title" rel="dct:type">Representation Learning for Syntactic and Semantic Theory</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="http://aaronstevenwhite.io" property="cc:attributionName" rel="cc:attributionURL">Aaron Steven White</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>. Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/aaronstevenwhite/representation-learning-course" rel="dct:source">https://github.com/aaronstevenwhite/representation-learning-course</a>.