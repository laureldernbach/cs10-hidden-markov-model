import java.io.*;
import java.util.*;
/**
 * Program to tag parts of speech in a text file given
 * previous tagged texts to calculate probability of a word's tag
 * based on previous tags of similar words/word types
 * 
 * @authors Laurel Dernbach
 * CS10 Spring 2019 PS5
 */
public class PatternRecognition {
	double U = -100; // unseen penalty
	static Map<String,TreeMap<String,Double>> transitionScores; // Map<currState, Map<nextState, transitionScore>>
	static Map<String,TreeMap<String,Double>> observationScores; //Map<observation, Map<state, observationScore>>
	
	
	public PatternRecognition() {
		transitionScores = new TreeMap<String, TreeMap<String, Double>>();
		observationScores = new TreeMap<String, TreeMap<String, Double>>();
	}
	
	/**
	 * @param text file of sentences, text file of corresponding tags
	 * populates maps created in constructor
	 */
	public void train(String wordsFilename, String tagFilename) throws IOException {
		BufferedReader words = new BufferedReader(new FileReader(wordsFilename));
		BufferedReader tags = new BufferedReader(new FileReader(tagFilename));
		transitionScores.put("start", new TreeMap<String,Double>());
		
		String wordsLine;
		while ((wordsLine = words.readLine())!= null) { 
			wordsLine.toLowerCase();
			String tagsLine = tags.readLine();
			String[] wordPieces = wordsLine.split(" ");
			String[] tagPieces = tagsLine.split(" ");
			
			for(int i=0; i<(wordPieces.length-1); i++) {
				// populate observationScores
				// if the current observation has an entry into observationScores, adjust scores
				if (observationScores.containsKey(wordPieces[i])) {
					if(observationScores.get(wordPieces[i]).containsKey(tagPieces[i])) {
						double newObsScore = (observationScores.get(wordPieces[i]).get(tagPieces[i]) + 1.0);
						observationScores.get(wordPieces[i]).put(tagPieces[i], newObsScore);
					}
					else {
						observationScores.get(wordPieces[i]).put(tagPieces[i], 1.0);
					}
				}
				// else, put the observation into map with score 1
				else {
					observationScores.put(wordPieces[i], new TreeMap<String, Double>());
					observationScores.get(wordPieces[i]).put(tagPieces[i], 1.0);
				}
				
				// populate transitionScores
				String currState = tagPieces[i];
				String nextState = tagPieces[i+1]; 
				//special start case (start transition)
				if (i == 0) {
					//System.out.println("i==0:" + currState);
					if (transitionScores.get("start").containsKey(currState)) {
						double newTransScore = (transitionScores.get("start").get(currState) + 1.0);
						transitionScores.get("start").put(currState, newTransScore);
					}
					else transitionScores.get("start").put(currState, 1.0);
					
					// add currState at i=0 to map
					if (transitionScores.containsKey(currState)) {
						if (transitionScores.get(currState).containsKey(nextState)) {
							double newTransScore = (transitionScores.get(currState).get(nextState) + 1.0);
							transitionScores.get(currState).put(nextState, newTransScore);
						}
						else {
							transitionScores.get(currState).put(nextState, 1.0);
						}
					}
					else {
						transitionScores.put(currState, new TreeMap<String, Double>());
						transitionScores.get(currState).put(nextState, 1.0);
					}
				}
				
				else { 
					if (transitionScores.containsKey(currState)) {
						if (transitionScores.get(currState).containsKey(nextState)) {
							double newTransScore = (transitionScores.get(currState).get(nextState) + 1.0);
							transitionScores.get(currState).put(nextState, newTransScore);
						}
						else {
							transitionScores.get(currState).put(nextState, 1.0);
						}
					}
					else {
						transitionScores.put(currState, new TreeMap<String, Double>());
						transitionScores.get(currState).put(nextState, 1.0);
					}
				}
			}
		}
		words.close();
		tags.close();	
	}
	
	/**
	 * converts observation and transition scores into log probabilities
	 * instead of whole numbers, so as to handle smaller numbers
	 */
	public void normalize() {
		// convert all scores to log scores
		for(String state:transitionScores.keySet()) {
			TreeMap<String,Double> stateMap = transitionScores.get(state);
			double totalTrans = 0;
			for(String tag:stateMap.keySet()) {
				totalTrans += stateMap.get(tag);
			}
			for(String tag2:stateMap.keySet()) {
				double prob = Math.log(stateMap.get(tag2) / totalTrans);
				transitionScores.get(state).put(tag2, prob);
			}
		}
		for(String word:observationScores.keySet()) {
			TreeMap<String,Double> wordMap = observationScores.get(word);
			double totalObs = 0;
			for(String obs:wordMap.keySet()) {
				totalObs += wordMap.get(obs);
			}
			for(String obs2:wordMap.keySet()) {
				double prob =  Math.log(wordMap.get(obs2) / totalObs);
				observationScores.get(word).put(obs2, prob);
			}
		}
	}
	
	/**
	 * @param text file with words
	 * @return string of tags for inputed text file based on training data
	 */
	public String viterbi(String filename) throws Exception {
		BufferedReader input = new BufferedReader(new FileReader(filename));
		String bestPathList = "";
		
		String line;
		while ((line = input.readLine())!= null) {
			List<String> observations = new ArrayList<String>(); // List<observation>
			List<Map<String,String>> observationBackTracks = new ArrayList<Map<String,String>>();
			line.toLowerCase();
			String[] words = line.split(" ");
			for(String word:words) {
				observations.add(word);
			}
			// set to avoid duplicates/heap space error
			Set<String> currStates = new HashSet<String>(); // List<state>
			currStates.add("start");
			
			Map<String, Double> currScores = new TreeMap<String,Double>(); // Map<state, currScore>
			currScores.put("start", 0.0);
			
			for(int i=0; i < observations.size(); i++) {
				// every observation has its own backtrack map
				Map<String,String> backTrack = new TreeMap<String,String>(); // Map<state, previous state>
				Set<String> nextStates = new HashSet<String>();
				Map<String, Double> nextScores = new TreeMap<String, Double>();
				
				// loop through every possible currState
			    for(String currState:currStates) {
			    	//  loop through every possible nextState for given currState
			    	if(transitionScores.containsKey(currState)) {
				    	for(String nextState:transitionScores.get(currState).keySet()) {
				    		nextStates.add(nextState);
				    		// check is word has been observed and grab score, if not apply penalty U
				    		double observationScore;
				    		if (observationScores.containsKey(observations.get(i))) {
					    		if (observationScores.get(observations.get(i)).containsKey(nextState)) {
						    		observationScore = observationScores.get(observations.get(i)).get(nextState);
					    		}
					    		else observationScore = U;
					    	}
					    	else observationScore = U;
				    		double nextScore = (currScores.get(currState)+
				    							transitionScores.get(currState).get(nextState)+
				    							observationScore);
		
				    		if(!nextScores.containsKey(nextState) || nextScore > nextScores.get(nextState)) {
				    			nextScores.put(nextState, nextScore);
				    			// remember that predecessor of nextState @ i is currState 
				    			backTrack.put(nextState, currState);
				    		}
				    	}
			    	}
			    }
			    observationBackTracks.add(backTrack);
			    currStates = nextStates;
			  	currScores = nextScores;
			} 
			
			// find highest score in currScores & remember the state that produced that score
			// search through all backTracks
			double highestScore = -1000000; // scores may be negative, start with very negative number so negatives closer to 0 are a higher score
			String highestState = null;
			for (String state:currScores.keySet()) {
				if (currScores.get(state) > highestScore) {
					highestScore = currScores.get(state);
					highestState = state;
				}
			}
			
			List<String> bestPath = new ArrayList<String>();
			String key = highestState;
			bestPath.add(key);
			for(int i=observationBackTracks.size()-1; i>=1; i--) {
				String state = observationBackTracks.get(i).get(key);
				bestPath.add(state);
				key = state;
			}
			
			Collections.reverse(bestPath);
			for(String item:bestPath) {
				bestPathList += (item + " ");
			}
			bestPathList += "\n";
		}
		input.close();
		System.out.println("tags for input: ");
		System.out.println(bestPathList);
		return bestPathList;
	}	
	
	/**
	 * @param string of tags (ideally returned by viterbi) and
	 * text file of tags that should be similar to tags
	 * 
	 * compares how similar the tags returned by viterbi are to the actual tags,
	 * prints to console correct and incorrect # of tags
	 */
	public void compare(String testTags, String tagsFile) throws IOException {
		BufferedReader input = new BufferedReader(new FileReader(tagsFile));
		int correct = 0;
		int incorrect = 0;
		String[] lines = testTags.split("\n");
		for(int i=0; i<=(lines.length-1); i++) {
			String[] tagsArray = lines[i].split(" ");
			String tags2 = input.readLine();
			String[] tags2Array = tags2.split(" ");
			for(int j=0; j<=(tagsArray.length-1); j++) {
				if (tagsArray[j].equals(tags2Array[j])) {
					correct += 1;
				}
				else incorrect += 1;
			}	
		}
		System.out.println("correct tags: " + correct);
		System.out.println("incorrect tags: " + incorrect);
	}
	
	public static void main(String [] args) throws Exception {
//		PatternRecognition test = new PatternRecognition();
//		test.train("src/train-test-sentances", "src/train-test-tags");
//		test.normalize();
//		System.out.println("trained transitions scores map:");
//		System.out.println(transitionScores);
//		System.out.println("trained observation scores map:");
//		System.out.println(observationScores + "\n");
//		test.viterbi("cat chase dog");
//		test.viterbi("dog watch cat chase dog");
//		test.viterbi("chase watch dog chase watch");
//		test.train("src/simple-train-sentences.txt", "src/simple-train-tags.txt");
//		test.normalize();
//		System.out.println("trained transitions scores map:");
//		System.out.println(transitionScores);
//		System.out.println("trained observation scores map:");
//		System.out.println(observationScores + "\n");
//		String tags = test.viterbi("src/simple-test-sentences.txt");
//		test.compare(tags, "src/simple-test-tags.txt");
		PatternRecognition brown = new PatternRecognition();
		brown.train("src/brown-train-sentences.txt", "src/brown-train-tags.txt");
		brown.normalize();
		String tags = brown.viterbi("src/brown-test-sentences.txt");
		brown.compare(tags, "src/brown-test-tags.txt");
	}
}