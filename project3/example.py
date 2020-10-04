def ent(df, attribute):
    target_variables = df.play.unique()  # This gives all 'Yes' and 'No'
    # This gives different features in that attribute (like 'Sweet')
    variables = df[attribute].unique()

    entropy_attribute = 0
    for variable in variables:
        entropy_each_feature = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute] == variable]
                      [df.play == target_variable])  # numerator
            den = len(df[attribute][df[attribute] == variable])  # denominator
            fraction = num/(den+eps)  # pi
            # This calculates entropy for one feature like 'Sweet'
            entropy_each_feature += -fraction*log(fraction+eps)
        fraction2 = den/len(df)
        # Sums up all the entropy ETaste
        entropy_attribute += -fraction2*entropy_each_feature

    return(abs(entropy_attribute))

def find_entropy(df):
    # To make the code generic, changing target variable class name
    Class = df.keys()[-1]
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy

def find_entropy_attribute(df, attribute):
    # To make the code generic, changing target variable class name
    Class = df.keys()[-1]
    target_variables = df[Class].unique()  # This gives all 'Yes' and 'No'
    # This gives different features in that attribute (like 'Hot','Cold' in Temperature)
    variables = df[attribute].unique()
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute] == variable]
                      [df[Class] == target_variable])
            den = len(df[attribute][df[attribute] == variable])
            fraction = num/(den+eps)
            entropy += -fraction*log(fraction+eps)
        fraction2 = den/len(df)
        entropy2 += -fraction2*entropy
    return abs(entropy2)

def find_winner(df):
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
        #         Entropy_att.append(find_entropy_attribute(df,key))
        IG.append(find_entropy(df)-find_entropy_attribute(df, key))
    return df.keys()[:-1][np.argmax(IG)]

def get_subtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)

def buildTree(df, tree=None):
    # To make the code generic, changing target variable class name
    Class = df.keys()[-1]

    # Here we build our decision tree
    # Get attribute with maximum information gain
    node = find_winner(df)

    # Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
    attValue = np.unique(df[node])

    # Create an empty dictionary to create tree
    if tree is None:
        tree = {}
        tree[node] = {}

    # We make loop to construct a tree by calling this function recursively.
    # In this we check if the subset is pure and stops if it is pure.
    for value in attValue:
        subtable = get_subtable(df, node, value)
        clValue, counts = np.unique(subtable['Eat'], return_counts=True)

        if len(counts) == 1:  # Checking purity of subset
            tree[node][value] = clValue[0]
        else:
            # Calling the function recursively
            tree[node][value] = buildTree(subtable)

    return tree
