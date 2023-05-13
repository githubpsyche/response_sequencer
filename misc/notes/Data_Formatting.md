I want to make a minimal dataset that is suitable for examining correspondence and eventually segmentation techniques.

Looking at the original code cell, what were they?

Identify the purpose of the original code. I measured how often the target unit matched a response that was most similar to that response as opposed to other responses.

What is the full correspondence problem? Given a set of response units and target units, measure how often the matching selected by the rater was the same as the matching selected by the automatic method.

So I'll need for all trials the response units and the target units.

Where do I already have response units? Inside coding_dfs, which the full_response column of the coding excel files. Same, of course, for source units. I only used tanscript trials for redundant validation.

But the pool of response units a target unit might be matched to is not just those that were matched to a target unit, but also those that weren't. It looks like transcript trials have also been segmented using line breaks. So I should use those along with the source unit texts. Do I need to do all of the same text preprocessing? Almost definitely not. But supposing I do, what code cells are needed? Most of them, yeah. The first four sections at minimum.

Well, no. Let's focus. A lot of these code cells are just for data validation. Let's see...

- First section reloads embeddings and identifies directories.  I'll need something like this.
- Second builds a list of paths and tags across all transcripts and coding files. I shouldn't have to do this every time but would require something like a unitary file to really compress. Loads coding DFS too.
- Third is focused on identifying stimulus and sense pool and extracting embeddings. I should only have to do this once and it's different between methods too.
- Thourth, all the transcript preprocessing is something I could sidestep by saving all_transcript_trials somewhere.
- Fifth preallocates for a data file appropriate for model evaluation. I don't need this for the unit matching project.
- Sixth is focused on coding recall order. Do I still needa ny of this? No.
- Seventh is the demo unit matching obtaining a 91% success rate.

Conclusions? 
- Loading data is universal, so I'll have something like 1/2 though it could require less code.
- sense pool and stimulus pool are stored as text files, so the stuff in 3 is included in data loading. Embeddings too and other details of my coding method will be imported/loaded. So 3 is inside 1/2
- If I save all transcripts, 4 is inside 1/2 as well.
- 5 is unnecessary 
- 6/7 is where I'd start coding.

So gist is if I save all_transcripts, then I'll just load data and start evaluating.

What's the general way to format evaluation data though?
Target units (in presentation order or prevalence order), response units (in response order), correspondences units (in response order, -1 for unmatched). I should probably also include the unsegmented trial response for developing segmentation schemes.

This is good enough progress. What's next? And what did I learn? I can get around gaming itch by playing videos of the thing I want to induldge and piping near where I should work. 