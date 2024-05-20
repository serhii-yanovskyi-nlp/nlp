import amrlib
stog = amrlib.load_stog_model()
graphs = stog.parse_sents(["The course of true love never did run smooth, for it encounters many obstacles.",
"All the world’s a stage, and all the men and women merely players, performing various roles.",
"Though she be but little, she is fierce and brave beyond measure.",
"A fool thinks himself to be wise, but a wise man knows himself to be a fool."])
for graph in graphs:
    print(graph)