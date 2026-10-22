# -*- coding: utf-8 -*-
import queue
import csv
import pandas as pd
import time


try:
    from Cleaner.Horizon.graph import Graph, topoSort, walk, tr
except ImportError:
    from .graph import Graph, topoSort, walk, tr


def parse_fd_rules(constrains_path):
    """
    Parse a rules file and extract functional dependency rules from the HORIZON_FD section.

    Supported formats:
    - Pure FD rules file: one rule per line formatted as "A => B" or "A ⇒ B".
    - Sectioned rules file: only rules under the [HORIZON_FD] section are read.

    :param constrains_path: path to the constraints file.
    :return: list of FD rules, each element is a (left_attr, right_attr) tuple.
    """
    fd_rules = []
    in_horizon_section = False
    has_sections = False

    with open(constrains_path, encoding='utf-8') as f:
        lines = f.readlines()

    # Check whether the file contains section markers
    for line in lines:
        if line.strip().startswith('['):
            has_sections = True
            break

    if has_sections:
        # Sectioned rules file: read only the [HORIZON_FD] section
        for line in lines:
            line = line.strip()
            if line.startswith('[HORIZON_FD]'):
                in_horizon_section = True
                continue
            elif line.startswith('[') and in_horizon_section:
                in_horizon_section = False
                continue

            if in_horizon_section and line and not line.startswith('#'):
                # Parse the FD rule, supporting both "=>" and "⇒"
                if '=>' in line:
                    parts = line.split('=>')
                elif '⇒' in line:
                    parts = line.split('⇒')
                else:
                    continue

                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    if left and right:
                        fd_rules.append((left, right))
    else:
        # Pure FD rules file: one rule per line
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse the FD rule, supporting both "=>" and "⇒"
            if '=>' in line:
                parts = line.split('=>')
            elif '⇒' in line:
                parts = line.split('⇒')
            else:
                continue

            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                if left and right:
                    fd_rules.append((left, right))

    return fd_rules


def BuildFDPatternGraph(D_path, constrains_path):
    """
    Build a functional dependency pattern graph from a data file and an FD constraints file.

    :param D_path: path to the CSV data file containing records of multiple attributes.
    :param constrains_path: path to the FD constraints file, formatted as "attr_A ⇒ attr_B".
    :return: the constructed graph object.
    """
    g = Graph()  # create the graph object
    my_dict = {}
    data = pd.read_csv(D_path)  # read the data file
    data = data.fillna("empty")  # fill missing values
    data = data.astype(str)  # convert data to strings for easier handling
    tot = len(data)  # total number of records

    # Parse the rules file using parse_fd_rules
    fd_rules = parse_fd_rules(constrains_path)

    if not fd_rules:
        raise ValueError(
            f"Failed to parse any FD rules from the rules file {constrains_path}. "
            f"Make sure the rules file contains a [HORIZON_FD] section or correctly "
            f"formatted FD rules."
        )

    # Process constraints and build an attribute-mapping dictionary that labels attributes
    # as LHS (left-hand side) or RHS (right-hand side).
    for left_attr, right_attr in fd_rules:
        if left_attr in my_dict and my_dict[left_attr] == 1:
            my_dict[right_attr] = 1
            continue
        my_dict[left_attr] = 0  # LHS attribute labeled as 0
        my_dict[right_attr] = 1  # RHS attribute labeled as 1

    # Add vertices and edges to the graph based on the constraints and data file
    for left_attr, right_attr in fd_rules:
        left_data = data[left_attr].tolist()  # values of the LHS attribute
        right_data = data[right_attr].tolist()  # values of the RHS attribute
        uni_left_data = list(set(left_data))  # deduplicated LHS values
        uni_right_data = list(set(right_data))  # deduplicated RHS values

        # Add vertices for the LHS attribute
        for item in uni_left_data:
            g.addVertex(str(item), left_attr, my_dict[left_attr])

        # Add vertices for the RHS attribute
        for item in uni_right_data:
            g.addVertex(str(item), right_attr, my_dict[right_attr])

        # Add edges representing functional dependencies
        for l_item, r_item in zip(left_data, right_data):
            g.addEdge(str(l_item), str(r_item))

    # Normalize edge weights (edge count divided by total record count)
    for v in g:
        for w in v.getConnections():
            v.connectedTo[w] = v.connectedTo[w] / tot

    # Enumerate edges and their weights (count only)
    cnt = 0
    for v in g:
        for w in v.getConnections():
            # print("( %s , %s ), %f" % (v.getId(), w.getId(), v.connectedTo[w]))
            cnt += 1

    return g  # return the constructed graph


def dfs(g, root, vis):
    """
    Depth-first search that computes the cumulative support and count starting from
    the root vertex.

    :param g: graph object.
    :param root: current vertex.
    :param vis: visit-flag dictionary recording which vertices have been visited.
    :return: cumulative support and count for the current vertex and its descendants.
    """
    if len(root.getConnections()) == 0:  # no outgoing edges -> return 0, 0
        return 0, 0
    if vis[root.attr] == 1:  # already visited -> skip (avoid revisiting / back edges)
        return 0, 0
    sup = 0  # cumulative support
    num = 0  # cumulative count
    vis[root.attr] = 1  # mark the current vertex as visited
    for v in root.getConnections():
        if vis[g.vertList[v.id].attr] == 0:  # if the neighbor has not been visited
            sup += root.connectedTo[v]  # accumulate the edge weight
            num += 1
            tmpa, tmpb = dfs(g, g.vertList[v.id], vis)  # recurse on the neighbor
            sup += tmpa
            num += tmpb
            root.connectedQLT[v] = (tmpa + root.connectedTo[v]) / (tmpb + 1)  # update quality score
    vis[root.attr] = 0  # clear the flag on backtracking
    return sup, num  # return cumulative support and count


# def dfs1(g, root, vis):
#     """
#     Depth-first search that prints the path and connection quality from root.
#
#     :param g: graph object
#     :param root: current vertex
#     :param vis: visit-flag dictionary
#     """
#     print(root.id)
#     if len(root.getConnections()) == 0:
#         return
#     if vis[root.attr] == 1:
#         return
#     vis[root.attr] = 1
#     for v in root.getConnections():
#         print(root.connectedQLT[v])
#         dfs1(g, g.vertList[v.id], vis)
#     vis[root.attr] = 0


def ComputePatternQulity(g):
    """
    Compute the pattern quality for every pattern in the graph.

    :param g: graph object.
    """
    vis = {}  # visit-flag dictionary tracking visit state of vertices
    for v in g:
        if v.getType() == 0:  # only process bound (LHS) vertices
            for vv in g:  # initialize visit flags
                vis[vv.attr] = 0
            dfs(g, v, vis)  # run DFS on the bound attribute to compute support and quality


def BuildSCCGraghAndSort(constrains_path):
    """
    Build the strongly connected component (SCC) graph from the FDs and topologically
    sort it.

    :param constrains_path: path to the constraints file. Entries are formatted as
        "attr_A ⇒ attr_B", meaning attribute A functionally depends on attribute B.
    :return:
        - order: topological order of the SCC graph.
        - tar: mapping from vertex to SCC id.
        - scc: list of SCCs.
        - G: adjacency-list representation of the original graph.
    """
    sccg = Graph()  # create the graph object
    G = {}  # adjacency-list representation

    # Parse the rules file using parse_fd_rules
    fd_rules = parse_fd_rules(constrains_path)

    if not fd_rules:
        raise ValueError(
            f"Failed to parse any FD rules from the rules file {constrains_path}."
        )

    # Build the functional dependency graph
    for left_attr, right_attr in fd_rules:
        sccg.addVertex(left_attr, "", 0)  # add LHS vertex
        sccg.addVertex(right_attr, "", 0)  # add RHS vertex
        sccg.addEdge(left_attr, right_attr)  # add FD edge (A -> B)

    # Convert the graph to adjacency-list form
    for v in sccg.vertList:
        tmp = set()
        for vv in sccg.vertList[v].getConnections():
            tmp.add(vv.id)  # collect neighbors
        G.update({v: tmp})  # update adjacency list

    # Compute the transpose graph
    GT = tr(G)

    # Compute topological order and strongly connected components
    seen = set()  # visited vertices
    scc = []  # stores SCCs
    for u in topoSort(G):  # topologically sort the graph
        if u in seen:
            continue
        C = walk(GT, u, seen)  # traverse in the transpose graph
        seen.update(C)  # update visited vertices
        scc.append(sorted(list(C.keys())))  # append the SCC to the result

    # Build the mapping from vertex to SCC id
    tar = {}
    cnt = 0
    for li in scc:
        for ui in li:
            tar.update({ui: cnt})
        cnt += 1

    # Build the SCC graph
    ret = {i: set() for i in range(cnt)}
    for li in G:
        for ui in G[li]:
            left = tar[li]  # SCC id of the source vertex
            right = tar[ui]  # SCC id of the target vertex
            if left != right:
                ret[left].add(right)

    # Compute in-degrees (used for topological sort)
    indegree = [0] * cnt
    for i in range(cnt):
        for j in ret[i]:
            indegree[j] += 1

    # Topological sort
    q = queue.Queue()
    for i in range(cnt):
        if indegree[i] == 0:  # enqueue vertices with zero in-degree
            q.put(i)
    order = []
    while not q.empty():
        ele = q.get()
        order.append(ele)
        for i in ret[ele]:
            indegree[i] -= 1
            if indegree[i] == 0:
                q.put(i)

    print(scc)
    print(ret)
    print(order)
    print(tar)
    return order, tar, scc, G  # return topological order, mapping, SCCs, and original graph


class TmpOrder:
    def __init__(self):
        """
        Initialize a TmpOrder object that represents a functional dependency item
        together with its ordering information.

        Attributes:
        - left: string, LHS (premise) attribute of the FD.
        - right: string, RHS (consequent) attribute of the FD.
        - lnum: int, SCC id of the LHS attribute.
        - rnum: int, SCC id of the RHS attribute.
        """
        self.left = ""  # LHS attribute of the FD
        self.right = ""  # RHS attribute of the FD
        self.lnum = 0  # SCC id of the LHS attribute
        self.rnum = 0  # SCC id of the RHS attribute


def OrderFDs(constrains_path, order, tar, scc, G):
    """
    Order the functional dependencies based on the SCCs and the topological sort.

    :param constrains_path: path to the constraints file.
    :param order: topological order of the SCC graph.
    :param tar: mapping from vertex to SCC id.
    :param scc: list of SCCs.
    :param G: adjacency-list representation of the original graph.
    :return: list of ordered functional dependencies.
    """
    OrderedFDs = []

    # Parse the rules file using parse_fd_rules
    fd_rules = parse_fd_rules(constrains_path)

    # Build FD objects
    for left_attr, right_attr in fd_rules:
        tmp = TmpOrder()
        tmp.lnum = tar[left_attr]  # SCC id of the LHS attribute
        tmp.rnum = tar[right_attr]  # SCC id of the RHS attribute
        tmp.left = left_attr  # LHS attribute
        tmp.right = right_attr  # RHS attribute
        OrderedFDs.append(tmp)

    # Sort by SCC ids
    OrderedFDs.sort(key=lambda x: (x.lnum, x.rnum))

    # Print the ordering
    for i in OrderedFDs:
        print(i.lnum, i.rnum)

    return OrderedFDs  # return the ordered FDs


# def export_res(pattern_expressions, dirty_path):
#     """
#     Export the repaired results to a CSV file.
#
#     :param pattern_expressions: repaired pattern expressions (repair results per tuple).
#     :param dirty_path: path to the dirty data file.
#     """
#     res_df = pd.read_csv(dirty_path)  # read the dirty data file
#
#     # Apply the repair results
#     for i in range(len(res_df)):
#         for v in pattern_expressions[i]:  # iterate over repair results in the pattern expression
#             res_df.iloc[i, list(res_df.columns).index(v)] = pattern_expressions[i][v]  # update the cell
#
#     # Save the repaired file
#     res_path = "./Repaired_res/horizon/" + task_name[:-1] + "/repaired_" + task_name + dirty_path[-25:-4] + ".csv"
#     res_df.to_csv(res_path, index=False)  # export the repaired data to CSV


def GeneratePatternPreservingRepairs(dirty_path, constraints_path, gt_wrong_cells, clean_df):
    """
    Generate pattern-preserving repairs by applying the FD pattern graph and the FD
    ordering to the dirty data.

    :param dirty_path: path to the dirty data file containing records to repair.
    :param constraints_path: path to the constraints file defining the FDs.
    :param gt_wrong_cells: list of ground-truth wrong cells, used to evaluate repairs.
    :param clean_df: DataFrame of clean (ground-truth) data used for reference.
    :return: list of pattern expressions containing the repaired data.
    """
    # 1. Build the FD pattern graph and compute the quality of every pattern
    g = BuildFDPatternGraph(dirty_path, constraints_path)
    ComputePatternQulity(g)

    # 2. Build the SCC graph and perform topological sorting
    order, tar, scc, G = BuildSCCGraghAndSort(constraints_path)

    # 3. Order the functional dependencies
    OrderedFDs = OrderFDs(constraints_path, order, tar, scc, G)

    pattern_expressions = []  # repaired pattern expressions
    rtable_set = []  # repair results per tuple

    # 4. Read the dirty data file
    with open(dirty_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, restval='empty')
        data = pd.read_csv(dirty_path)
        data = data.fillna("empty")  # fill missing values with 'empty'
        data = data.astype(str)  # ensure all values are strings

        for i in range(len(data)):
            Rtable = {}  # repair results for the current tuple
            check = 0  # flag used to track repair status

            # Initialize Rtable: for bound attributes in the graph, add the original value
            for v in g.vertList:
                if g.vertList[v].type == 0:  # only bound (LHS) attributes
                    content = data.loc[i, g.vertList[v].attr]  # original value
                    Rtable.update({g.vertList[v].attr: content})  # update Rtable

            # 5. Iterate over the ordered FDs to perform repair
            for j in range(len(OrderedFDs)):
                # If Rtable does not contain the LHS of the dependency, add it
                if OrderedFDs[j].left not in Rtable.keys():
                    Rtable.update({OrderedFDs[j].left: data.loc[i, OrderedFDs[j].left]})

                Lval = Rtable[OrderedFDs[j].left]  # value of the LHS attribute
                # print('Lval', Lval)
                # If the RHS attribute is already in Rtable, skip
                if OrderedFDs[j].right in Rtable:
                    continue

                if Lval == '':  # treat empty LHS values as 'empty'
                    Lval = 'empty'
                # if Lval == 'empty':  # skip if LHS is 'empty'
                #     continue
                # Select the best repair value based on connection quality in the graph
                maxedge = -1
                for v in g.vertList[Lval].getConnections():
                    # if v.id=='empty':
                    #     continue
                    # print(v.id + "," + str(g.vertList[Lval].connectedQLT[v]))
                    if v.attr == OrderedFDs[j].right and g.vertList[Lval].connectedQLT[v] > maxedge:
                        maxedge = g.vertList[Lval].connectedQLT[v]
                        maxp = v.id  # record repair candidate
                        # print('maxp', maxp)

                # If a valid repair candidate is found, update the RHS attribute
                if maxedge != -1 and maxp != 'empty':
                    Rtable.update({OrderedFDs[j].right: maxp})

                # If the pattern perfectly matches the ground truth, process the repair
                if PERFECTED:
                    if (i, list(clean_df.columns).index(OrderedFDs[j].right)) not in gt_wrong_cells:
                        Rtable.update({OrderedFDs[j].right: data.loc[i, OrderedFDs[j].right]})
                    if (i, list(clean_df.columns).index(OrderedFDs[j].left)) not in gt_wrong_cells:
                        Rtable.update({OrderedFDs[j].left: data.loc[i, OrderedFDs[j].left]})

            # Append the repaired Rtable to the result
            pattern_expressions.append(Rtable)
            rtable_set.append(Rtable)

        # print(rtable_set)

    return pattern_expressions  # return the pattern expressions (repaired data)


def dirty_cells(dirty_file, clean_file):
    """
    Identify dirty cells by comparing the dirty data and the clean data.

    :param dirty_file: dirty-data DataFrame.
    :param clean_file: clean-data (ground truth) DataFrame.
    :return: list of dirty cells; each element is an (i, j) tuple indicating that the
             cell at row i, column j is erroneous.
    """
    dirty_c = []  # list of dirty cells
    for i in range(len(clean_file)):  # iterate over rows
        for j in range(len(clean_file.columns)):  # iterate over columns
            if dirty_file.iloc[i, j] != clean_file.iloc[i, j]:  # mismatch with ground truth
                dirty_c.append((i, j))  # append dirty cell index (i, j)
    return dirty_c  # return the list of dirty cells

PERFECTED = 0


def Horizon(dirty_path, rule_path, clean_path):
    """
    Main entry point that runs the data cleaning process and returns the repaired
    pattern expressions.

    :param dirty_path: path to the dirty CSV file containing data to be repaired.
    :param rule_path: path to the constraints rules file containing FD rules.
    :param clean_path: path to the clean CSV file used as ground truth.
    :return: repaired pattern expressions.
    """
    start_time = time.time()  # record the start time

    # Read dirty and clean data
    dirty_df = pd.read_csv(dirty_path).astype(str)  # read dirty data as strings
    clean_df = pd.read_csv(clean_path).astype(str)  # read clean data as strings
    dirty_df = dirty_df.fillna("empty")  # fill missing values in dirty data with "empty"
    clean_df = clean_df.fillna("empty")  # fill missing values in clean data with "empty"

    # Identify dirty cells
    dirty_c = dirty_cells(dirty_df, clean_df)  # find dirty cells
    gt_wrong_cells = [(i, j) for i in range(len(clean_df)) for j in range(len(clean_df.columns))
                      if clean_df.iloc[i, j] != dirty_df.iloc[i, j]]  # ground-truth dirty cells

    # Generate pattern-preserving repairs
    pattern_expressions = GeneratePatternPreservingRepairs(dirty_path, rule_path, gt_wrong_cells, clean_df)
    end_time = time.time()  # record the end time

    return pattern_expressions, dirty_c, end_time - start_time


