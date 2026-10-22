import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Tuple
# import streamlit as st

def sort_opinfo_by_weights(grouped_opinfo_dict, group_weights):
    """Sort the grouped operator information by weight."""
    sorted_dict = {}

    for level in grouped_opinfo_dict:
        nodes = grouped_opinfo_dict[level]
        for targetNode in nodes:
            weights = group_weights[targetNode]
            weighted_ops = []
            for cleaner in nodes[targetNode]:
                weighted_ops.append((cleaner, weights[nodes[targetNode].index(cleaner)]))

            sorted_ops = sorted(weighted_ops, key=lambda x: x[1], reverse=True)
            sorted_dict[targetNode] = sorted_ops

    return sorted_dict

# 6. Node levels

def compute_node_levels(graph: nx.DiGraph) -> Dict[str, int]:
    """Level (maximum depth) of every node of the graph."""
    levels = {node: 0 for node in graph}
    for node in nx.topological_sort(graph):
        levels[node] = max([levels[pred] + 1 for pred in graph.predecessors(node)], default=0)
    return levels


# 7. Hierarchical layout
def hierarchical_layout(graph: nx.DiGraph, vertical_gap: float = 1.0, horizontal_gap: float = 2.0) -> Dict[str, Tuple[float, float]]:
    """Hierarchical layout of the graph."""
    levels = compute_node_levels(graph)
    level_groups = {}
    for node, level in levels.items():
        level_groups.setdefault(level, []).append(node)
    pos = {}
    for level, nodes in level_groups.items():
        for i, node in enumerate(sorted(nodes)):
            pos[node] = (i * horizontal_gap, -level * vertical_gap)
    return pos
def draw_hierarchical_dependency_graph(dependency_graph, analysis):
    """
    Draw the hierarchical dependency graph.

    Parameters:
    dependency_graph : networkx.DiGraph
        dependency graph
    analysis : dict
        analysis result, including the list of start nodes

    Output:
    the drawn dependency graph
    """
    plt.figure(figsize=(15, 11))
    pos = hierarchical_layout(dependency_graph, vertical_gap=2, horizontal_gap=2)  # Layout parameters
    # Nodes
    start_nodes = [node for comp in analysis.values() for node in comp["Starting_Nodes"]]
    end_nodes = [node for node in dependency_graph.nodes() if node not in start_nodes]
    nx.draw_networkx_nodes(dependency_graph, pos, nodelist=end_nodes, node_color='skyblue', node_size=4000)
    # Label font
    nx.draw_networkx_labels(dependency_graph, pos, font_size=32)
    # Highlight the start nodes
    nx.draw_networkx_nodes(dependency_graph, pos, nodelist=start_nodes, node_color='lightgreen', node_size=3000)
    # Edges
    nx.draw_networkx_edges(dependency_graph, pos, arrowstyle='->', arrowsize=60, edge_color='gray', width=5)
    plt.title('Hierarchical Dependency Graph', fontsize=20)
    plt.axis('off')
    # plt.show()
    return plt

def getSingle_opinfo(singles):
    """Information of the single-attribute cleaners."""
    singleDomain = []
    singlesInfo = []
    for single in singles:
        singleop = dict()
        singleDomain.append(single.domain)
        singleop['name'] = single.name
        singleop['type'] = 'single'
        singleop['attr'] = single.domain
        singleop['format'] = str(single.format)
        singlesInfo.append(singleop)
    return singlesInfo


def cleaner_grouping(processing_order, models):
    """Group the cleaners by processing order and model."""
    groups = {}
    for level_index, level in enumerate(processing_order):
        print(f"\nlevel {level_index + 1}, nodes: {level}")
        group = {}
        for node in level:
            if node in models and models[node]:
                group[node] = models[node]
        if group:  # Add only non-empty groups
            groups[level_index] = group
    return groups


def generate_plantuml_corrected(singleCleaners, groupCleaners, dependencies):
    """PlantUML text of the execution order of the cleaners."""
    plantuml_text = "@startuml\n"
    plantuml_text += "!define RectOperator(x) class x << (O, orchid) >>\n\n"
    plantuml_text += "skinparam rectangle {\n"
    plantuml_text += "    BackgroundColor<<O>> LightBlue\n"
    plantuml_text += "    BorderColor Black\n"
    plantuml_text += "    ArrowColor Black\n"
    plantuml_text += "}\n\n"
    plantuml_text += "title Coarse-grained Cleaner Execution Order Recommendation\n\n"
    plantuml_text += "' layout settings removed\n"
    plantuml_text += "hide empty description\n"
    plantuml_text += "hide empty methods\n"
    plantuml_text += "hide empty fields\n\n"

    single_end = "SINGLE_END"
    plantuml_text += f"RectOperator({single_end})\n\n"
    last_single = ''
    for i in range(len(groupCleaners)):
        plantuml_text += f"RectOperator(level_{i+1}_end)\n\n"
    for group, cleaners in groupCleaners.items():
        for targetNode, cleaner_list in cleaners.items():
            for index, cleaner in enumerate(cleaner_list):
                cleanername = cleaner.name
                plantuml_text += f"RectOperator({cleanername})\n"
                plantuml_text += f"{cleanername} : source={''.join(cleaner.source)}\n"
                plantuml_text += f"{cleanername} : target={''.join(cleaner.target)}\n"
    for index, cleaner in enumerate(singleCleaners):
        plantuml_text += f"RectOperator({cleaner['name']})\n"
        plantuml_text += f"{cleaner['name']} : {cleaner['attr']}\n"
        plantuml_text += f"{cleaner['name']} : {cleaner['format']}\n"
        # Single-attribute cleaners connect to SINGLE_END
        if last_single:
            plantuml_text += f"{last_single} <-right-> {cleaner['name']} : parallel\n"
        last_single = f"{cleaner['name']}"
        plantuml_text += f"{last_single} -down-> {single_end} : next\n"

    first_group_Cleaner = single_end
    for level, groups in groupCleaners.items():
        for targetNode, _ in groups.items():
            sorted_cleaners = dependencies[targetNode]
            plantuml_text += f"{first_group_Cleaner} -down-> {sorted_cleaners[0][0].name} : next\n"
            for i in range(1, len(sorted_cleaners)):
                plantuml_text += f"{sorted_cleaners[i - 1][0].name} -down-> {sorted_cleaners[i][0].name} : next\n"
            plantuml_text += f"{sorted_cleaners[len(sorted_cleaners)-1][0].name} -down-> level_{level+1}_end : next\n"
        first_group_Cleaner = f"level_{level+1}_end"
    plantuml_text += "@enduml"
    return plantuml_text


def explain_cleaners_process(cleaners, dependencies):
    """Explain the processing order of the cleaners."""
    explanation = "Process Explanation:\n\n"
    explanation += "**1. Single-attribute cleaners** (cleaner type as 'single') are executed first and are parallel to each other. Specifically, this includes:\n"
    single_cleaners = [cleaner["name"] for cleaner in cleaners if cleaner["type"] == "single"]
    for single in single_cleaners:
        explanation += f"   - {single}\n"
    explanation += "Once these single-attribute cleaners are completed, they move to the next stage, collectively marked as 'SINGLE_END'.\n\n"

    explanation += "**2. Multi-attribute cleaners** (cleaner type as 'multi') are executed according to their categories, with cleaners within each category following a specific order, and the categories themselves are parallel to each other. The specific categories and their execution order are as follows:\n"
    for I, group in enumerate(dependencies, start=1):
        explanation += f"   Category {I}:\n"
        for cleaner_name in group:
            explanation += f"      - {cleaner_name}\n"
        explanation += "   Within this category, cleaners are executed in the order listed above.\n\n"

    explanation += "**3. The first cleaner of each multi-attribute cleaner category is connected to the 'SINGLE_END' node**, indicating that these multi-attribute cleaner categories will start executing in parallel after the completion of single-attribute cleaners.\n\n"
    explanation += "**4. The first cleaner of multi-attribute cleaner categories are connected in parallel to each other**, indicating they will start executing simultaneously.\n"

    return explanation


# def explain_cleaners_process_multilangu(cleaners, dependencies):
#     """Explain the cleaner processing workflow"""
#
#     # language configuration
#     EXPLAIN_LANGUAGES = {
#         'en': {
#             'process_explanation': "Process Explanation:\n\n",
#             'single_cleaners_intro': "**1. Single-attribute cleaners** (cleaner type as 'single') are executed first and are parallel to each other. Specifically, this includes:\n",
#             'single_end_explanation': "Once these single-attribute cleaners are completed, they move to the next stage, collectively marked as 'SINGLE_END'.\n\n",
#             'multi_cleaners_intro': "**2. Multi-attribute cleaners** (cleaner type as 'multi') are executed according to their categories, with cleaners within each category following a specific order, and the categories themselves are parallel to each other. The specific categories and their execution order are as follows:\n",
#             'category': "Category",
#             'within_category': "Within this category, cleaners are executed in the order listed above.\n\n",
#             'first_cleaner_connection': "**3. The first cleaner of each multi-attribute cleaner category is connected to the 'SINGLE_END' node**, indicating that these multi-attribute cleaner categories will start executing in parallel after the completion of single-attribute cleaners.\n\n",
#             'parallel_execution': "**4. The first cleaner of multi-attribute cleaner categories are connected in parallel to each other**, indicating they will start executing simultaneously.\n"
#         },
#         'zh': {
#             'process_explanation': "Process:\n\n",
#             'single_cleaners_intro': "**1. Single-attribute cleaners** (type 'single') run first, in parallel:\n",
#             'single_end_explanation': "After the single-attribute cleaners, the next stage starts at 'SINGLE_END'.\n\n",
#             'multi_cleaners_intro': "**2. Multi-attribute cleaners** (type 'multi') run by category, in a fixed order inside a category and in parallel across categories:\n",
#             'category': "Category",
#             'within_category': "Inside this category the cleaners run in the order listed above.\n\n",
#             'first_cleaner_connection': "**3. The first cleaner of each multi-attribute category connects to 'SINGLE_END'**, so these categories start in parallel after the single-attribute cleaners.\n\n",
#             'parallel_execution': "**4. The first cleaners of the multi-attribute categories are connected in parallel**, so they start at the same time.\n"
#         }
#     }
#
#     def get_explain_text(key):
#         """explanation text of the current language"""
#         current_lang = st.session_state.get('language', 'en')
#         return EXPLAIN_LANGUAGES[current_lang].get(key, key)
#
#     # build the explanation text
#     explanation = get_explain_text('process_explanation')
#     explanation += get_explain_text('single_cleaners_intro')
#
#     single_cleaners = [cleaner["name"] for cleaner in cleaners if cleaner["type"] == "single"]
#     for single in single_cleaners:
#         explanation += f"   - {single}\n"
#     explanation += get_explain_text('single_end_explanation')
#
#     explanation += get_explain_text('multi_cleaners_intro')
#     for I, group in enumerate(dependencies, start=1):
#         explanation += f"   {get_explain_text('category')} {I}:\n"
#         for cleaner_name in group:
#             explanation += f"      - {cleaner_name}\n"
#         explanation += f"   {get_explain_text('within_category')}"
#
#     explanation += get_explain_text('first_cleaner_connection')
#     explanation += get_explain_text('parallel_execution')
#
#     return explanation
