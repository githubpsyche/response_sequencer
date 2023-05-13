Suppose I start with the MAT file as my starting point. What do I add and what do I remove?

pres_itemids identifies the sense units that are located in another file. that's cool; this is a solid "target_units" entry and i may even rename the field. but probably not.

then response units? I presently directly code correspondences -- either as orderings of target units or direct identification of the target units' string index.

I'll also need some text representation of each response unit and at minimum id-based identfication of these in my MAT file.

That's enough for automating correspondences. Given a set of response units and target units, match them how the people did.

But how do I test accounts of segmentation? Could store the transcript per trial along with vectors containing starting and ending position of each identified unit. Hmm. Let's clarify this actually. How do I plan to evaluate segmentation schemes? I've only clarified for myself their importance.

How do I compare two sets of extracted response units? It's possible to look at low-level statistics. Average length of a unit. 

Stuff like the jaccard index offer a way to evaluate the similarity of two units. I measure the area of overlap -- here, the number of characters shared by the two units -- and divide by the area of unit -- the number of characters in at least one unit. This is 1 when units match and 0 when there is no overlap at all. What about when one contains the other? Area of overlap is the size of the smaller. Area of union is the size of the larger. So still less than 1. Ok.

How does this work when I have multiple units?

Work me through an example. Suppose I am segmenting a text with 50 characters into 2 segments. My first method sets a boundary at indices 25. My second method sets a boundary at index 30.

Jaccard similarity for first unit is... 25/30. For second unit is 20/25.
(.83 + .8) / 2 = .815.

The further these two segments get, the worse similarity gets.

But how do I match units? I guess I search for a minimal average jaccard similarity.  For example, set at 15 or 30.

15/30 and 20/35. 
.5 + .57 / 3 = .535

A different way to evaluate agreement is to look at the number of characters that the methods disagree about. Eg, 5 or 15 in the above examples. 
5/50 = 1/10 = 10%. 15/50 = 3/10 = 30%

How do I generalize to multiple segmentations?

15/30 vs 25, for example.

Find the smallest possible disagreement. So match first 15 to first 25, and last 20 to last 25. Leaves a disagreement about...15 characters again? Ok, sure, this sounds reasonable. How do I explain this?

Adjusted rand metric is my metric. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html

https://en.wikipedia.org/wiki/Rand_index

Key terms are "partition" and "data clustering". 

But how do I use when clusterings are only partial? I guess I can only consider the character indices that are identfied by a rater as belonging to a cluster. I can measure whether the method also differentiates between these sets of characters.

Other evaluation techniques here: https://en.wikipedia.org/wiki/Cluster_analysis

Purity is the name for the method I imagined. For each cluster, find the number of data points from the most common class in the cluster. The mean of this over all clusters is a measure of the exten tto which clusters "contain a single class".

This measure doesn't penalize having many clusters, and more clusters will make it easier to produce a high purity. A purity score of 1 is always possible by putting each data point in its own cluster. So purity is nice, but I don't know why I want it.

Fowlkes–Mallows index apparently addresses some esoteric limitations of the rand index.

Hmm, I see. The Rand index article actually provides a framework for identifying the true positives, false positives, true negatives, and false negatives in a partitioning given a gold standard. So I can use Jaccard or other metrics too.

Okay, I have a decent understanding of my options now. What does this mean for data representation? The basic unit for evaluation is a vector of class labels for each character in the text. This representation isn't easy to look at.

I think rather than looking for a more derived representation, I'll just code a function that can transform into the format preferred for clustering analysis evaluation. In that context, I'll still need the transcript text for each trial and to be sure that every response unit is a subset of the transcript text. It looks like I never enforce this in the senses dataset. I probably don't enforce it in the narrative recall datasets either. Hmm. When I can't do a direct matching, I guess I'll have to use my maximum similarity method to ensure a unit is selected. Maybe to differentiate I should separately record start and end points for each unit after all. Ah, but this assumes every unit will be contiguous! Nah, I do need sklearn's representation after all. Ugh, but what about overlapping units? I'll need a breaking rule. Coding recall order requires some nonoverlap between units.

So next steps:
1. Add full response text to representation
2. Add clustering labels for each character in text using rater units
3. Add some tracking of text representation for each respone unit (so I can compute embeddings, etc)

This is enough for segmentation.

But back to correspondences. How do I track rejection performance if I only have recalled response units? How do I evaluate decision not to match a target unit to a response unit at all without adding segmentation dependency?



more skleran metrics: https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation

