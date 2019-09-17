# CONTES/HONOR

This installation guide enable the use of two methods of normalization with concepts from a domain-specific ontology: CONTES and HONOR through the corpus processing engine [AlvisNLP/ML](https://bibliome.github.io/alvisnlp/).
For more details on the methods, see related paper:
- [CONTES](http://www.aclweb.org/anthology/W17-2312) 
- [HONOR](https://www.aclweb.org/anthology/L18-1543)


## Intallation
1. Get AlvisNLP/ML from GitHub:
```
$ git clone https://github.com/Bibliome/alvisnlp.git
```

2. Follow the instructions from:
- For Linux OS: [https://github.com/Bibliome/alvisnlp/blob/master/README.md](https://github.com/Bibliome/alvisnlp/blob/master/README.md)
- For Win OS: [https://github.com/Bibliome/alvisnlp/blob/master/WINDOWS.md](https://github.com/Bibliome/alvisnlp/blob/master/WINDOWS.md)

Comments:
- Be sure that you are root/administrator user.
- (Optional) If you wish to apply some specific preprocessing to your corpus: get [TreeTagger](https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/) tool.
- You can jump the "Default command-line options" and "Web service" sections.

3. In your base directory of AlvisNLP/ML, fill the CONTES (and TreeTagger if installed) modules in the `default-param-values.xml` file:
```
<module class="fr.inra.maiage.bibliome.alvisnlp.bibliomefactory.modules.treetagger.TreeTagger">
	<treeTaggerExecutable>|||pathToYourTreeTaggerExecutable|||</treeTaggerExecutable>
	<parFile>|||pathToYourTreeTaggerDirectory|||\lib\english-utf8.par</parFile>
	<inputCharset>UTF-8</inputCharset>
	<outputCharset>UTF-8</outputCharset>
</module>

<module class="fr.inra.maiage.bibliome.alvisnlp.bibliomefactory.modules.contes.Word2Vec">
	<contesDir>|||pathToYourContesDirectory|||</contesDir>
	<workers>|||DefaultNumberOfWorkers|||</workers>
</module>

<module class="fr.inra.maiage.bibliome.alvisnlp.bibliomefactory.modules.contes.ContesTrain">
	<contesDir>|||pathToYourContesDirectory|||</contesDir>
</module>

<module class="fr.inra.maiage.bibliome.alvisnlp.bibliomefactory.modules.contes.ContesPredict">
	<contesDir>|||pathToYourContesDirectory|||</contesDir>
</module>
```

4. If you want to use an AlvisNLP/ML plan, add this parameters in your command line:
```
$ -inputDir . -inputDir ???
```
For instance:
```
$ alvisnlp -inputDir . -inputDir |||yourAlvisDirectoryPath|||
```

## CONTES 

Ongoing...

## HONOR 

Ongoing...

