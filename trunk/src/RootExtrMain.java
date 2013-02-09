// You have to add weka.jar and libsvm.jar to the build path in order to compile.
// ALL ARABIC TEXT MUST BE IN UTF-8, input files, etc
// this file is in utf-8
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;
import java.util.StringTokenizer;
import java.util.regex.Pattern;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.StringReader;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;

public class RootExtrMain {
	
	// CONSTANTS
	public static final String FATHA = "َ";
	public static final String DAMMA = "ُ";
	public static final String KASRA = "ِ";
	public static final String SHADDA = "ّ";
	public static final String ALIF = "ا";
	public static final String HAMZMAD = "آ";
	public static final String HAMZA = "ء";
	public static final String[] DIACS_arr = {"َ","ً","ُ","ٌ","ِ","ٍ","ّ","ْ"};
	public static final String DIACS_str = "ًٌٍَُِّْ";
	public static final String DIACS_regex = "[ًٌٍَُِّْ]";
	public static final int l = 3;
	public static final char group[][] = { 	
		{'ا','إ','ت','م'},
		{'س','ن'},
		{'ت','ا','و','ي'},
		{'ا','ن','و','ي','ة'},
		{'ا','ن','ة'},
		{'ا','ة'},
		{0}
	};
	
	// to avoid recompiling regex's every iteration, we define them here 
	public static Pattern regex_newline = Pattern.compile("\\n");
	public static Pattern regex_hamza = Pattern.compile("[ء-ئ]");
	public static Pattern regex_letterShadda = Pattern.compile("([ء-ي]"+SHADDA+")");
	public static Pattern regex_shadda = Pattern.compile(SHADDA);
	public static Pattern regex_hamzmad = Pattern.compile("آ");
	public static Pattern regex_allLetters = Pattern.compile("[ء-ي]");
	public static Pattern regex_kashidaDiac = Pattern.compile("ـ([َُِْ])");
	public static Pattern regex_allDiacs = Pattern.compile(DIACS_regex);
	public static Pattern regex_wawYa= Pattern.compile("[وي]");
	//return regex_.matcher(string).replaceAll("replace by");
	
	
	public static void main(String[]args) throws Exception {
		LibSVM[] svms = new LibSVM[l];
		boolean csvF = false; // determines format used.. false=>is arff
		
		// setting class attribute 
		// Create vector to hold nominal values "first", "second", "third"
		FastVector my_nominal_values = new FastVector(3);
		my_nominal_values.addElement("pos0");
		my_nominal_values.addElement("pos1");
		my_nominal_values.addElement("pos2");
		my_nominal_values.addElement("pos3");
		my_nominal_values.addElement("pos4");
		my_nominal_values.addElement("pos5");
		my_nominal_values.addElement("pos6");
		my_nominal_values.addElement("pos7");
		my_nominal_values.addElement("unclassified");

		// Create nominal attribute "position"
		Attribute position = new Attribute("position", my_nominal_values);			

		// =======================READ========================
		/* <------- '//*' to train.. '/*' to skip training 
		// if you have the model already trained from 
		// a previous run. the model can be found as 3 files
		// with extension .model.
		// 
		// ===================================================
		
		//===========================
		// FEATURE EXTRACTION
		//===========================

		// Read raw words from csv file previously cleaned by some regex operations
		String rfName =  "مصادر_ثلاثية.csv";
		//String rfName =  "inMicro.csv";
		Scanner scnr = new Scanner (new File(rfName));
		ArrayList<String> sroots = new ArrayList<String>(30000); // this is too expensive and perhaps stupid..
		ArrayList<String> sderivs = new ArrayList<String>(30000); 
		int countLines = 0;
		while(scnr.hasNextLine()) {
			String tmp = scnr.nextLine().trim();
			countLines++;
			StringTokenizer x = new StringTokenizer(tmp, ",");
			if(x.countTokens() != 2) {
				System.out.println("Format error. "+rfName+":"+countLines+"\n  "+tmp);
				scnr.close();
				System.exit(0);
			}
			sroots.add(x.nextToken());
			sderivs.add(x.nextToken());
		}
		scnr.close();
		String[] roots = (String[]) sroots.toArray(new String[sroots.size()]);	
		String[] derivs = (String[]) sderivs.toArray(new String[sderivs.size()]);
		if(roots.length != derivs.length) {
			System.out.println("Input Format error. Num of roots different than num of derivs. How did this happen ?!!!");
			System.exit(0);
		}
		
		// some preprocessing
		// not needed anymore.. done in featExtract
//		for(int i=0; i<roots.length; i++) {
//			roots[i] = hamzaNorm(roots[i]); // normalize hamza
//			derivs[i] = hamzaNorm(derivs[i]); // normalize hamza
//			derivs[i] = shaddaSub(derivs[i]); // resolve shadda
//			derivs[i] = hamzmadSub(derivs[i]);// resolve hamzmad آ
//		}
		
		
		// do feature extraction and
		// save the features to files
		String[] ffNames = {"feat1.arff","feat2.arff","feat3.arff"};			
		if(csvF)
			ffNames = new String[] {"feat1.csv","feat2.csv","feat3.csv"};
		if(l!=ffNames.length) {
			System.out.println("Wait wait... WHAT?!! Let me stop you here because, eventually, something will go wrong. I though we're dealing with 3 classifiers. I have "+l+" and "+ffNames.length);
			System.exit(0);
		}
		File[] files = new File[l];
		for(int i=0; i<l; i++) {
			files[i] = new File(ffNames[i]);
		}
		PrintWriter[] outs = new PrintWriter[l];
		for(int i=0; i<l; i++) {
			outs[i] = new PrintWriter(files[i]);
		}
		
		// print datasets to files
		String[] feats;
		feats = getFeats(derivs, roots, csvF);
		for(int i=0; i<l; i++) {
			outs[i].print(feats[i]);
			outs[i].close();
		} 
		
		
		//===========================
		// AFTER FEATURE EXTRACTION
		// TRAIN SVM
		//===========================
		
		// Read Data.. three datasets one for each letter of the root => three classifiers  
		DataSource[] sources = new DataSource[l];
		Instances[] data = new Instances[l];
		for(int i=0; i<l; i++) {
			sources[i] = new DataSource(ffNames[i]);
			data[i] = sources[i].getDataSet();
		}
		
		// setting class attribute if the data format does not provide this information
		for(int i=0; i<l; i++) {
			if(csvF) data[i].setClass(position);
			if (data[i].classIndex() == -1) {
				data[i].setClassIndex(data[i].numAttributes()-1);
			}
		}
		 
		// setup parameters for SVM
		String[] params = weka.core.Utils.splitOptions("-S 0 -K 2 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.001 -P 0.1");
		
		
		// Evaluate SVM (optional to test your features and parameters)
		// evaluate using 10-fold.. test parameters
		// setup parameters.. see report for choice of parameters  
//		LibSVM svmEval = new LibSVM();
//		svmEval.setOptions(params);
//		Evaluation eval = new Evaluation(data[2]);
//		eval.crossValidateModel(svmEval, data[2], 10, new Random(1));
//		System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		
		// setup the three classifiers.. use the same parameters from evaluation
		for(int i=0; i<l; i++) {
			svms[i] = new LibSVM();
			svms[i].setOptions(params);
			svms[i].buildClassifier(data[i]);
		}
		
		//*/
		
		// load/save the trained models
		String[] modelfn = new String[l];
		for(int i=0; i<l; i++)
			modelfn[i] = "svm"+(i+1)+".model";
		if(svms[0]==null) { 
			System.out.println("Loading models"); 
			for(int i=0; i<l; i++) {
				svms[i] = (LibSVM) weka.core.SerializationHelper.read(modelfn[i]);
				System.out.println("Loaded svm model from "+modelfn[i]);
			}			
		} else { // save them 
			System.out.println("Saving models"); 
			for(int i=0; i<l; i++) {
				weka.core.SerializationHelper.write(modelfn[i], svms[i]);
				System.out.println("Saved svm model in "+modelfn[i]);
			}			
		}
		//*
		//===========================
		//  AFTER Training
		//  CLASSIFY unseen instances
		//===========================
		// classify new instances.. use the trained classifiers
		// load unclassified data, note that everything will be done in threes because we have three feature sets
		// deriv is the string to be analyzed
		String deriv = "استفهام";
		String[] fl = {"temp1", "temp2", "temp3"};
		File[] tempFile = new File[l];
		for(int i=0; i<l; i++) {
			tempFile[i] = new File(fl[i]);
		}
		
		// get the features of our input
		//String[] csvFeats = getCsvFeats(deriv);
		//String[] arffFeats = getArffFeats(deriv);
		String[] unclsdFeats = getFeats(deriv, csvF);
		// convert them to ARFF
		// if the feats are in CSV format, they must be written
		// into a text file and then read again.. unfortunately,
		// this is how weka works; it is explicitly stated in the
		// documentation. 
		Instances[] unclsd = new Instances[l];
		for(int i=0; i<l; i++) {
			if(csvF) {
			CSVLoader loader = new CSVLoader();
			loader.setSource(new ByteArrayInputStream(unclsdFeats[i].getBytes("UTF-8")));
			unclsd[i] = loader.getDataSet();
		    ArffSaver saver = new ArffSaver();
		    saver.setInstances(unclsd[i]);
		    saver.setFile(tempFile[i]);
		    saver.writeBatch();
		    unclsd[i] = new Instances(
                    new BufferedReader(
                      new FileReader(fl[i])));
			} else {
			    unclsd[i] = new Instances(
	                    new BufferedReader(
	                      new StringReader(unclsdFeats[i])));
			}
		}

		// setting class attribute if the data format does not provide this information
		for(int i=0; i<l; i++) {
			if(csvF) unclsd[i].setClass(position);
			if (unclsd[i].classIndex() == -1) {
				unclsd[i].setClassIndex(unclsd[i].numAttributes()-1);
			}
		}
		 
		// create copy
		Instances clsd[] = new Instances[l];
		for(int i=0; i<l; i++)
			clsd[i] = new Instances(unclsd[i]);
		 
		// label instances (classify)
		for(int i=0; i<l; i++) {
			double clsLabel=0;
			for (int j=0; j<unclsd[i].numInstances(); j++) {
				clsLabel = svms[i].classifyInstance(unclsd[i].instance(j));
				clsd[i].instance(j).setClassValue(clsLabel);
			}
		}
		
		// show final output (only one instance is handled)
		int classIdxs[] = {0,0,0};
		char letters[] = {0,0,0};
		for(int i=0; i<l; i++) {
			classIdxs[i] = (int) clsd[i].instance(0).classValue();
			String label = classLabels[i].trim().split(",")[classIdxs[i]];
			try {
				int realPos = Integer.parseInt(label.substring(label.length()-1))-1;
				System.out.println(classIdxs[i]+": "+label+" -> charAt("+realPos+")");
				letters[i] = removeDiacs(deriv).charAt(realPos);
			} catch (java.lang.StringIndexOutOfBoundsException e) {
				letters[i] = '?';
			}
			// i think this should be shadda-unfolded not just vocals-removed
			// WARNING string operation not try-catched
		}
		String root = new String(letters);
		System.out.println("SVM says: "+deriv+" is derived from "+root);
		System.out.println("Guesser says: "+guessDerived(deriv,root));
		
		
		// delete temp files.. not working probably because ArffSaver and unclsd have not freed them
		for(int i=0; i<l; i++) {			
			tempFile[i].delete();
		}
		//*/
	}
	//===========================
	//      END OF MAIN
	//===========================
	
	// returns the features of a given word in CSV or ARFF format
	private static String[] getFeats(String deriv, boolean csvF)  throws FileNotFoundException {
		if(csvF)
			return getCsvFeats(deriv);
		else
			return getArffFeats(deriv);
	}

	// returns the features of a given word in ARFF format
	private static String[] getArffFeats(String deriv)  throws FileNotFoundException {
		// append the header
		String[]ss = getHeader(false);

		// append the numbers
		String[] f = featExtract(deriv);
		for(int i=0; i<ss.length; i++)
			ss[i] += f[i];
		return ss;
	}

	// returns the features of a given word in CSV format
	private static String[] getCsvFeats(String deriv)  throws FileNotFoundException {
		String[] h = featNamesWClassCsv;
		String[] f = featExtract(deriv);
		if(h.length != f.length) System.out.println("HOW DID THIS HAPPEN ?!! DEBUG NOW");
		String[]ss = new String[h.length];
		for (int i=0; i<ss.length; i++) {
			ss[i] = h[i]+f[i];
		}
		return ss;
	}

	// returns the features of a given array of words in CSV or ARFF format
	private static String[] getFeats(String[] derivs, boolean csvF)  throws FileNotFoundException {
		return getFeats(derivs, new String [derivs.length], csvF);
	}

	// returns the features of a given array of words in ARFF format
	private static String[] getArffFeats(String[] derivs, String[] roots) throws FileNotFoundException {
		return getFeats(derivs, roots, false);
	}

	// returns the features of a given array of words in CSV or ARFF format
	private static String[] getFeats(String[] derivs, String[]roots, boolean csvF)  throws FileNotFoundException {
		// append the header
		String[]sss = getHeader(csvF);

		// append the numbers
		for (int j=0; j<derivs.length; j++) { // for each deriv 
			String[] f = featExtract(derivs[j],roots[j]);
			for(int i=0; i<sss.length; i++) // for 3
				sss[i] += f[i];
		}
		return sss;
	}
	
	// returns the feature file header for CSV or ARFF formats
	private static String[] getHeader(boolean csvF) {
		if(csvF) {
			String[] h = featNamesWClassCsv;
			String[]header = new String[h.length];
			for (int i=0; i<header.length; i++) {
				header[i] = h[i];
			}	
			return header;
		} else { //ARFF
			String[] h = featNamesCsv;
			String[]header = new String[h.length];
			for (int i=0; i<header.length; i++) {
				header[i] = "% Features of derived words to root letter "+(i+1)+"\n%\n";
				header[i] += "@RELATION rootL"+(i+1)+"\n\n";
				String[] attribs = h[i].trim().split(",");
				for (int j=0; j<attribs.length; j++)
					header[i] +="@ATTRIBUTE "+attribs[j]+"\tNUMERIC\n";
				header[i] +="@ATTRIBUTE position\t{"+classLabels[i]+"}\n";
				header[i] +="\n@DATA\n";
			}	
			return header;			
		}
	}
	private static String[] classLabels = {
		"pos0,pos1,pos2,pos3,pos4",
		"pos0,pos2,pos3,pos4,pos5",
		"pos0,pos3,pos4,pos5,pos6,pos7"
	};
	private static String[] featNamesCsv = {
		"length,isLetter1InCorrespondingGroup,isLetter2InCorrespondingGroup,isLetter3InCorrespondingGroup,isLetter4InCorrespondingGroup,WhatLetter1Haraka,WhatLetter2Haraka,WhatLetter3Haraka,WhatLetter4Haraka,isLetter1Vowel,isLetter2Vowel,isLetter3Vowel,isLetter4Vowel,isLetter1Hamza,isLetter2Hamza,isLetter3Hamza,isLetter4Hamza\n",
		"length,isLetter2InCorrespondingGroup,isLetter3InCorrespondingGroup,isLetter4InCorrespondingGroup,isLetter5InCorrespondingGroup,WhatLetter2Haraka,WhatLetter3Haraka,WhatLetter4Haraka,WhatLetter5Haraka,isLetter2Vowel,isLetter3Vowel,isLetter4Vowel,isLetter5Vowel,isLetter2Hamza,isLetter3Hamza,isLetter4Hamza,isLetter5Hamza,rootL1Position\n",
		"length,isLetter3InCorrespondingGroup,isLetter4InCorrespondingGroup,isLetter5InCorrespondingGroup,isLetter6InCorrespondingGroup,isLetter7InCorrespondingGroup,WhatLetter3Haraka,WhatLetter4Haraka,WhatLetter5Haraka,WhatLetter6Haraka,WhatLetter7Haraka,isLetter3Vowel,isLetter4Vowel,isLetter5Vowel,isLetter6Vowel,isLetter7Vowel,isLetter3Hamza,isLetter4Hamza,isLetter5Hamza,isLetter6Hamza,isLetter7Hamza,rootL1Position,rootL2Position\n"
	};
	
	private static String[] featNamesWClassCsv = {
		"length,isLetter1InCorrespondingGroup,isLetter2InCorrespondingGroup,isLetter3InCorrespondingGroup,isLetter4InCorrespondingGroup,WhatLetter1Haraka,WhatLetter2Haraka,WhatLetter3Haraka,WhatLetter4Haraka,isLetter1Vowel,isLetter2Vowel,isLetter3Vowel,isLetter4Vowel,isLetter1Hamza,isLetter2Hamza,isLetter3Hamza,isLetter4Hamza,position\n",
		"length,isLetter2InCorrespondingGroup,isLetter3InCorrespondingGroup,isLetter4InCorrespondingGroup,isLetter5InCorrespondingGroup,WhatLetter2Haraka,WhatLetter3Haraka,WhatLetter4Haraka,WhatLetter5Haraka,isLetter2Vowel,isLetter3Vowel,isLetter4Vowel,isLetter5Vowel,isLetter2Hamza,isLetter3Hamza,isLetter4Hamza,isLetter5Hamza,rootL1Position,position\n",
		"length,isLetter3InCorrespondingGroup,isLetter4InCorrespondingGroup,isLetter5InCorrespondingGroup,isLetter6InCorrespondingGroup,isLetter7InCorrespondingGroup,WhatLetter3Haraka,WhatLetter4Haraka,WhatLetter5Haraka,WhatLetter6Haraka,WhatLetter7Haraka,isLetter3Vowel,isLetter4Vowel,isLetter5Vowel,isLetter6Vowel,isLetter7Vowel,isLetter3Hamza,isLetter4Hamza,isLetter5Hamza,isLetter6Hamza,isLetter7Hamza,rootL1Position,rootL2Position,position\n"
	};
	
	// Extracts 3 feature vectors from the input string, one for each classifier
	// not every vector of the three is unique in features. For example, all three have
	// a feature "length" (length of deriv)
	// deriv is the word we want to find the root for.
	// This method should not be called directly
	private static String[] featExtract(String deriv) throws FileNotFoundException {
		return featExtract(deriv,null);
	}
	private static String[] featExtract(String deriv, String root) throws FileNotFoundException {
		
		// preprocess input string
		// ---------------------------
		String root_mod=null;
		boolean training = false;
		if(root!=null) {
			root_mod=hamzaNorm(root);
			training = true;
		}
		String deriv_mod = deriv;
		deriv_mod = hamzaNorm(deriv_mod); // normalize hamza
		deriv_mod = shaddaSub(deriv_mod); // resolve shadda
		deriv_mod = hamzmadSub(deriv_mod);// resolve hamzmad آ
		// separate the string from its diacritical/vocalization/tashkeel marks  
		String noT = removeDiacs(deriv_mod); // the input string without Tashkeel
		String T = getDiacs(deriv_mod); // a string of only the Tashkeel of the input
		// DONE preprocess
		//----------------------------
		// prepare output
		// featVs is an array of 3 feature vectors each corresponding to a classifier.
		String featVs[] = {"","",""};
		int m=7; // max num of letters expected
		// ignore if length > m
		if(noT.length() > m) {
			// TODO warning hardcoded
			String featVsDummy[] = {
					"0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,pos0\n",
					"0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,pos0\n",
					"0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,pos0\n"
					};
			System.out.println("Warning: ("+deriv+") words longer than "+m+" letters are not supported. returned dummy feature vector.");
			return featVsDummy; // should cause no trouble to caller
		}
		
		// setup the vars to be distributed among the three feature vectors
		int[] isLetterInGroup = new int[m];
		for(int i=0; i<m; i++) 
			isLetterInGroup[i] = isLetterInGroup(noT,i,i);
		int[] WhatLetterHaraka= new int[m];
		for(int i=0; i<m; i++) 
			WhatLetterHaraka[i] = WhatLetterHaraka(T,i);
		int[] isLetterVowel= new int[m];
		for(int i=0; i<m; i++) 
			isLetterVowel[i] = isLetterVowel(noT,i);
		int[] isLetterHamza= new int[m];
		for(int i=0; i<m; i++) 
			isLetterHamza[i] = isLetterHamza(noT,i);
		String[] cls = new String[l];
		int[] clsInt = new int[l]; // initialized to zeros by java
		// TODO should I fill it with -1 for "unclassified" to keep 0 for "not found" ?
		// I think not because in testing, I don't want to be surprised by a -1 !!
		// plus it won't make a difference, because clsInt will take a new value very soon
		//java.util.Arrays.fill(clsInt, -1); 
		if(root_mod!=null && training) { 
			clsInt = getInstanceClassInt(noT, root_mod);
			for(int i=0; i<l; i++) { 
				cls[i] = "pos"+clsInt[i];
			}
		}
		else
			for(int i=0; i<l; i++) 
				// TODO Don't know if the unclassified instances can be 
				// labeled by the default label.
				// should be ok.. it is only used for new instances 
				// that will directly go into the classifier.
				//cls[i]="unclassified";
				cls[i]="pos0";
		
		int j,f,t; // feat vector num, from, to
		
		// feature vector 1.. concerned letters (1-4)
		j=0; f=0; t=3;
		featVs[j] += noT.length()+",";
		for(int i=f; i<=t; i++) 
			featVs[j] += (isLetterInGroup[i]+",");
		for(int i=f; i<=t; i++) 
			featVs[j] += (WhatLetterHaraka[i]+",");
		for(int i=f; i<=t; i++) 
			featVs[j] += (isLetterVowel[i]+",");
		for(int i=f; i<=t; i++) 
			featVs[j] += (isLetterHamza[i]+",");
		featVs[j] += cls[j];
		featVs[j] += "\n";
		
		// feature vector 2.. concerned letters (2-5)
		j=1; f=1; t=4;
		featVs[j] += noT.length()+",";
		for(int i=f; i<=t; i++) 
			featVs[j] += (isLetterInGroup[i]+",");
		for(int i=f; i<=t; i++) 
			featVs[j] += (WhatLetterHaraka[i]+",");
		for(int i=f; i<=t; i++) 
			featVs[j] += (isLetterVowel[i]+",");
		for(int i=f; i<=t; i++) 
			featVs[j] += (isLetterHamza[i]+",");
		// TODO in case of classifying an unclassified instance (and testing as well),
		// this should be updated with the value the previous classifier found..
		// so.. how to do it ???
		featVs[j] += (clsInt[j-1]+",");
		featVs[j] += cls[j];
		featVs[j] += "\n";
		
		// feature vector 3.. concerned letters (3-7)
		j=2; f=2; t=6;
		featVs[j] += noT.length()+",";
		for(int i=f; i<=t; i++) 
			featVs[j] += (isLetterInGroup[i]+",");
		for(int i=f; i<=t; i++) 
			featVs[j] += (WhatLetterHaraka[i]+",");
		for(int i=f; i<=t; i++) 
			featVs[j] += (isLetterVowel[i]+",");
		for(int i=f; i<=t; i++) 
			featVs[j] += (isLetterHamza[i]+",");
		featVs[j] += (clsInt[j-2]+",");
		featVs[j] += (clsInt[j-1]+",");
		featVs[j] += cls[j];
		featVs[j] += "\n";
				
		//System.out.println("Feats for "+deriv+" ("+noT+" + "+T+"), "+root+"\n"+featVs[0]+featVs[1]+featVs[2]);
		return featVs;
	}
	
	// returns the class for this instance as an integer. used for setting up the training data
	private static int[] getInstanceClassInt(String deriv, String root) {
		//System.out.println("finding pos: \nroot:"+root+"\nderiv: "+deriv);
		int[] classes = new int[l];
		int pos = deriv.indexOf(root.charAt(0)) + 1; // +1 because 0 is reserved for 'not found'  
		if (pos>4) pos=0;
		classes[0]= pos; 
		pos = (locateLetter(root.charAt(1), deriv, 1)+1);
		if(pos>5) pos=0;
		classes[1]= pos; 
		classes[2]= locateLetter(root.charAt(2), deriv, 2)+1; 
		return classes;		
	}
	// returns the class for this instance as a string for weka demands so. 
	// used for setting up the training data
	private static String[] getInstanceClass(String deriv, String root) {
		String[] classes = new String[l];
		int[] x = getInstanceClassInt(deriv, root);
		if(x.length!=classes.length) {
			System.err.println("What is happening here ?!!");
			System.exit(0);
		}
		for(int i=0; i<x.length; i++)
			classes[i] = "pos"+x[i];
		return classes;
	}
	
	// normalizes hamza replacing all forms by one
	private static String hamzaNorm(String string) {
		return regex_hamza.matcher(string).replaceAll("ء");		
	}

	// substitue a shadda by its equivalent
	//unfold shadda: a letter with shadda is equivalnet to two instances of the same letter, where the first has a sukoon
	private static String shaddaSub(String string) {
		string = regex_letterShadda.matcher(string).replaceAll("$1ْ$1");
		string = regex_shadda.matcher(string).replaceAll("");
		return string;
	}
	
	// hamzmad (آ) replace by equivalent (hamza+mad) (ءا)
	private static String hamzmadSub(String string) {
		return regex_hamzmad.matcher(string).replaceAll("ءا");
	}
	
	// normalize vowel letters
	private static String vowelNorm(String string) {
		return regex_wawYa.matcher(string).replaceAll("ا");
	}
	
	// returns string of diacs with indexes matching with letters they modify 
	private static String getDiacs(String string) {
		// if character is letter followed by diac, replace by diac
		// else replace by kashida
		// shadda should already be unfolded
		// tanween is not accounted for
		string = regex_allLetters.matcher(string).replaceAll("ـ");
		string = regex_kashidaDiac.matcher(string).replaceAll("$1");
		return string;
	}
	
	// returns the word unvocalized (undiacritized)
	public static String removeDiacs(String string) {
		return regex_allDiacs.matcher(string).replaceAll("");
	}
	
	// is the character a diacritical sign/ a vocalization mark
	public static boolean isDiac(char c) {
		return DIACS_str.indexOf(c) > -1;
	}

	private static int WhatLetterHaraka(String T, int i) {
		// we want zero to mean don't care.. but how
		try { 
			if(  T.charAt(i) == 'َ' )
				return 1;
			else if(T.charAt(i) == 'ِ')
				return 2;
			else if(T.charAt(i) == 'ُ')
				return 3;
			else if(T.charAt(i) == 'ْ')
				return 4;
		}
		catch(Exception e ) { return 0; }
		return 0;
	}

	private static int locateLetter(char toLocate, String toSearch, int from, int to) {
		try { 
			int loc = toSearch.substring(from,to).indexOf(toLocate)+from; 
			if (loc == -1) {
				
			}
			return loc; 
		} catch (StringIndexOutOfBoundsException e) {
			return -1;
		}
	}

	private static int locateLetter(char toLocate, String toSearch, int from) {
		int loc = toSearch.indexOf(toLocate, from);
		if(loc == -1) return loc;
		else return loc;
	}

	private static int isLetterVowel(String noT, int i) {
		try { if(  noT.charAt(i) == 'ا'
				|| noT.charAt(i) == 'و'
				|| noT.charAt(i) == 'ي'
				|| noT.charAt(i) == 'ى')
			return 1; }
	catch(Exception e ) { return -1; }
		 return 0;
	}

	private static int isLetterHamza(String noT, int i) {
		try { if(  noT.charAt(i) == 'ء'
				|| noT.charAt(i) == 'ئ'
				|| noT.charAt(i) == 'ؤ'
				|| noT.charAt(i) == 'أ'
			    || noT.charAt(i) == 'إ')
			return 1; }
		catch(Exception e ) { return -1; }
		 return 0;
	}

	private static int isLetterInGroup(String s, int i, int j) {
		try { for (int x=0; x<group[j].length; x++) {
			if(s.charAt(i)==group[j][x])
				return 1; 
		} }
		catch(Exception e ) { return -1; }
		return 0;
	}
	
	// a guesser to return true if deriv is likely derived from root
	public static boolean guessDerived(String deriv, String root) {
		// deriv is the word to be tested whether it is derived from root
		//Text Normalization
		root  = hamzaNorm(root);// normalize hamza
		deriv = hamzaNorm(deriv);// normalize hamza
		root  = vowelNorm(root); // normalize vowels
		deriv = vowelNorm(deriv); // normalize vowels
		deriv = shaddaSub(deriv); //unfold shadda
		deriv = removeDiacs(deriv); //remove tashkeel
		deriv = hamzmadSub(deriv); // equiv آ
		
		// The guesser
		boolean success = false;
		for(int i=0; i<root.length()-1; i++) {
			try {
				int ti1 = deriv.indexOf(root.charAt(i));
				int ti2 = deriv.indexOf(root.charAt(i+1)); 
				if(ti1 < ti2 && ti1 > -1) 
					success = true;
				else
					return false;
			} catch (StringIndexOutOfBoundsException e) {return false;}
		}
		return success;
	}

}

