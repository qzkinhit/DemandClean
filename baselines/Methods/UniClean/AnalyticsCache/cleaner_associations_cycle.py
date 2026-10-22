from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set
import networkx as nx

# from AnalyticsCache.get_cleaner_excute_info import draw_hierarchical_dependency_graph


def PreParamClassier(cleanners):
    # Edge list and the two cleaner classes
    edges = []  # Edges
    single = []  # Single-domain cleaners
    multi = []  # Multi-domain cleaners
    # Classify the cleaners
    for cleaner in cleanners:
        if isinstance(cleaner.domain, set):
            # A cleaner whose domain is a set is a multi-domain cleaner
            multi.append(cleaner)
            for t in cleaner.target:
                s = cleaner.source.copy()
                if t in s:
                    s.remove(t)
                edges.append((s, [t]))
        else:
            # Otherwise it is a single-domain cleaner
            single.append(cleaner)
    return edges, single, multi


# 1. Dependency graph
def build_dependency_graph(edges: List[Tuple[List[str], List[str]]]) -> nx.DiGraph:
    """Dependency graph from tuple pairs."""
    graph = nx.DiGraph()
    for source, target in edges:
        for s in source:
            for t in target:
                graph.add_edge(s, t)
    return graph

# 2. Key nodes
def identify_key_points(graph: nx.DiGraph) -> List[str]:
    """Key nodes with in-degree 0."""
    return [node for node in graph if graph.in_degree(node) == 0]

# 3. Merge the nodes on cycles
def merge_cycles(graph: nx.DiGraph) -> nx.DiGraph:
    """Merge the nodes on cycles of the graph."""
    node_mapping = {node: {node} for node in graph.nodes}
    while True:
        try:
            cycles = list(nx.simple_cycles(graph))
        except nx.NetworkXNoCycle:
            break
        if not cycles:
            break
        for cycle in cycles:
            cycle_nodes = set(cycle)
            merged_node = ','.join(sorted(cycle_nodes))
            node_mapping[merged_node] = set()
            for node in cycle_nodes:
                node_mapping[merged_node].update(node_mapping.pop(node))
            # Create a new graph with the merged node
            new_graph = nx.DiGraph()
            for node in graph.nodes:
                if node not in cycle_nodes:
                    new_graph.add_node(node)
            new_graph.add_node(merged_node)
            for u, v in graph.edges:
                if u in cycle_nodes and v in cycle_nodes:
                    continue
                if u in cycle_nodes:
                    new_graph.add_edge(merged_node, v)
                elif v in cycle_nodes:
                    new_graph.add_edge(u, merged_node)
                else:
                    new_graph.add_edge(u, v)
            graph = new_graph
            break
    return graph, node_mapping

# 4. Partition the graph at the key nodes
def partition_graph(graph: nx.DiGraph, key_points: List[str]) -> List[Tuple[List[str], str]]:
    """Partition the graph at the key nodes."""
    partitions = []
    for key in key_points:
        dependent_nodes = nx.descendants(graph, key) | {key}
        for node in dependent_nodes:
            if graph.out_degree(node) > 0:
                targets = list(graph.successors(node))
                partitions.append((targets, node))
    return partitions

# 5. Processing order
def determine_processing_order(partitions: List[Tuple[List[str], str]]) -> List[Tuple[List[str], str]]:
    """Processing order."""
    return sorted(partitions, key=lambda x: x[1])



# 8. Connected components
def find_connected_components(graph: nx.DiGraph) -> List[nx.DiGraph]:
    """Split the graph into connected components."""
    return [graph.subgraph(c).copy() for c in nx.weakly_connected_components(graph)]

# 9. Topological order of the components, cycle detection
def topological_sort_with_cycles(graph: nx.DiGraph) -> Tuple[List[str], bool]:
    """Topological order of a connected component, with cycle detection."""
    try:
        return list(nx.topological_sort(graph)), False  # No cycle
    except nx.NetworkXUnfeasible:
        return [], True  # Cycle detected

# 10. Start nodes of every component
def find_starting_nodes(graph: nx.DiGraph, sorted_nodes: List[str]) -> List[str]:
    """Start nodes of every connected component."""
    in_degree = graph.in_degree()
    return [node for node in sorted_nodes if in_degree[node] == 0]

# 11. Dependency analysis of the components
def analyze_dependencies(edges: List[Tuple[List[str], List[str]]]) -> Dict[str, Dict]:
    """Analyse the dependencies and return the information of the components."""
    dependency_graph = build_dependency_graph(edges)
    dependency_graph, node_mapping = merge_cycles(dependency_graph)
    connected_components = find_connected_components(dependency_graph)
    analysis = {}
    for i, component in enumerate(connected_components):
        order, has_cycle = topological_sort_with_cycles(component)
        start_nodes = find_starting_nodes(component, order) if not has_cycle else []
        analysis[f"Component_{i + 1}"] = {
            "Topological_Order": order,
            "Has_Cycle": has_cycle,
            "Starting_Nodes": start_nodes
        }
    return analysis, node_mapping

# 12. Processing order in the target format
def format_combined_processing_order(processing_order: List[Tuple[List[str], str]], node_mapping: Dict[str, Set[str]]) -> List[Tuple[List[Set[str]], List[Set[str]]]]:
    """Processing order in the target format, sources of the same target merged."""
    target_to_sources = {}
    for targets, source in processing_order:
        for target in targets:
            target_to_sources.setdefault(target, set()).add(source)
    formatted_order = []
    for target, sources in target_to_sources.items():
        formatted_order.append((list(map(lambda x: node_mapping[x], sources)), [node_mapping[target]]))
    return sorted(formatted_order, key=lambda x: len(x[0]))

# 13. Analysis and visualization of the dependencies
def analyze_and_visualize_dependencies(edges: List[Tuple[List[str], List[str]]]):
    """Analyse and visualize the dependencies."""
    analysis, node_mapping = analyze_dependencies(edges)
    dependency_graph = build_dependency_graph(edges)
    dependency_graph, node_mapping = merge_cycles(dependency_graph)
    key_points = identify_key_points(dependency_graph)
    partitions = partition_graph(dependency_graph, key_points)
    processing_order = determine_processing_order(partitions)
    format_processing_order = format_combined_processing_order(processing_order, node_mapping)
    # draw_hierarchical_dependency_graph(dependency_graph, analysis)
    return analysis, format_processing_order
def analyze_and_visualize_dependencies_save_plt(edges: List[Tuple[List[str], List[str]]]):
    """Analyse and visualize the dependencies."""
    analysis, node_mapping = analyze_dependencies(edges)
    dependency_graph = build_dependency_graph(edges)
    dependency_graph, node_mapping = merge_cycles(dependency_graph)
    key_points = identify_key_points(dependency_graph)
    partitions = partition_graph(dependency_graph, key_points)
    processing_order = determine_processing_order(partitions)
    format_processing_order = format_combined_processing_order(processing_order, node_mapping)
    # plt=draw_hierarchical_dependency_graph(dependency_graph, analysis)
    plt=-1
    return analysis, format_processing_order,plt
# Readable description of the analysis
def explain_analysis_results(analysis: Dict[str, Dict], processing_order: List[Tuple[List[Set[str]], List[Set[str]]]]) -> str:
    """Readable text that explains the analysis and the processing order."""
    explanation = ""

    # Explain the analysis
    for component, details in analysis.items():
        explanation += f"\n{component}:\n"
        explanation += f"  - Topological Order: {', '.join(details['Topological_Order'])}\n"
        explanation += f"  - Has Cyclic Dependencies: {'Yes' if details['Has_Cycle'] else 'No'}\n"
        explanation += f"  - Starting Nodes: {', '.join(details['Starting_Nodes'])}\n"

    # Explain the processing order
    explanation += "\nProcessing Order:\n"
    for source_set, target_set in processing_order:
        source_str = ' and '.join([','.join(sorted(s)) for s in source_set])
        target_str = ' and '.join([','.join(sorted(t)) for t in target_set])
        explanation += f"  - When {source_str} are ready, {target_str} can be processed\n"

    return explanation

def extract_source_target_sets(processing_order: List[Tuple[List[Set[str]], List[Set[str]]]]):
    """Source and target sets of the processing order."""
    source_sets = []
    target_sets = []
    for sources, targets in processing_order:
        source_sets.append(sources)
        target_sets.append(targets)
    return source_sets, target_sets

def explain_analysis_results_zh(analyze: Dict[str, Dict],
                                processing_order: List[Tuple[List[Set[str]], List[Set[str]]]]):
    """Readable text that explains the analysis and the processing order."""
    explanation = "Analysis:\n"
    for component, data in analyze.items():
        explanation += f"{component} topological order: {data['Topological_Order']}\n"
        explanation += f"{component} has a cycle: {'yes' if data['Has_Cycle'] else 'no'}\n"
        explanation += f"{component} start nodes: {data['Starting_Nodes']}\n"
    explanation += "\nProcessing order:\n"
    for sources, targets in processing_order:
        explanation += f"sources: {sources} -> targets: {targets}\n"
    return explanation

def discover_cleaner_associations(AttrDependencies):
    """Analyse and visualize the attribute dependencies."""
    # Analyse and visualize the attribute dependencies
    analyze, processing_order,plt = analyze_and_visualize_dependencies_save_plt(AttrDependencies)
    # Source and target sets
    source_sets, target_sets = extract_source_target_sets(processing_order)
    # Explain the result
    explain = explain_analysis_results_zh(analyze, processing_order)

    return source_sets, target_sets, explain, processing_order,plt




def associations_classier(multi, source_sets: List[List[Set[str]]], target_sets: List[List[Set[str]]]):
    """Classify the multi-domain cleaners."""
    # Graph and in-degree table
    graph = defaultdict(list)
    indegree = defaultdict(int)
    all_nodes = set()
    targets_in_use = set()  # Nodes that occur in target_sets

    # Fill graph and in-degree table, record the targets
    for sources, targets in zip(source_sets, target_sets):
        for target in targets:
            target_node = ','.join(sorted(target))
            targets_in_use.add(target_node)  # Record the target
            all_nodes.add(target_node)
            for source in sources:
                source_node = ','.join(sorted(source))
                all_nodes.add(source_node)
                graph[source_node].append(target_node)
                indegree[target_node] += 1

    # Keep the nodes that occur in target_sets only
    filtered_nodes = {node for node in all_nodes if node in targets_in_use or any(
        node == ','.join(sorted(list(x))) for x in [item for sublist in source_sets for item in sublist])}

    # Topological levels that can run in parallel
    queue = deque([node for node in filtered_nodes if indegree[node] == 0])
    levels = []  # Parallel levels
    nodes = []
    # Levels with the nodes before merging
    while queue:
        current_level = []
        current_nodes = []
        for _ in range(len(queue)):
            node = queue.popleft()
            current_level.append(set(node.split(',')))
            current_nodes.append(node)
            for adjacent in graph[node]:
                if adjacent in filtered_nodes:  # Only the filtered nodes
                    indegree[adjacent] -= 1
                    if indegree[adjacent] == 0:
                        queue.append(adjacent)
        levels.append(current_level)
        nodes.append(current_nodes)

    # Models of the filtered result
    models = defaultdict(list)  # A list, so that cleaners can be added
    for cleaner in multi:
        for target in cleaner.target:  # cleaner.target can hold several targets
            target_node = target
            for level in levels:
                for group in level:
                    if target_node in group:
                        models[','.join(sorted(group))].append(cleaner)
                        break

    return levels, dict(models), nodes  # Back to a plain dictionary for the output