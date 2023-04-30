from typing import Dict, List, Any
import pandas as pd

class Message:
    def __init__(self, node_id, author, timestamp, parent_id,
                 topic, discussion_id, text, discussion_stance_id, rebuttal, thread_id, branch):
        self.node_id = node_id
        self.author = author
        self.timestamp = timestamp
        self.parent_id = parent_id
        self.topic = topic
        self.discussion_id = discussion_id
        self.text = text
        self.discussion_stance_id = discussion_stance_id
        self.rebuttal = rebuttal
        self.thread_id = thread_id
        self.branch = branch

    def __repr__(self):
        return f"Message(node_id={self.node_id}, author={self.author}, timestamp={self.timestamp}, parent_id={self.parent_id})"

class ConversationNode:
    def __init__(self, message: Message, children: List["ConversationNode"] = None):
        self.message = message
        self.children = children or []

    def add_child(self, child: "ConversationNode"):
        self.children.append(child)

    def init_children(self, uninitialized_node_ids, all_nodes, root_node):

        for node_id in uninitialized_node_ids:
            if (node_id == root_node.message.node_id):
                child_node_id = node_id
                child_node = all_nodes.get(child_node_id)
                self.add_child(child_node)

        if len(uninitialized_node_ids) < len(all_nodes)-1:
            print("removing: ", self.message.node_id)
            uninitialized_node_ids.remove(self.message.node_id)
            assert self.message.node_id not in uninitialized_node_ids

        if len(uninitialized_node_ids) == 0:
            return
        else:
            next_node = all_nodes.get(uninitialized_node_ids[0])
            next_node.init_children(uninitialized_node_ids, all_nodes)


class Conversation:
    def __init__(self, messages: List[Message]):
        self.root = None
        self.nodes = {}

        self.participants = []
        self.messages = messages
        '''init all nodes'''
        for message in messages:
            node = ConversationNode(message)
            self.participants.append(message.author)
            self.nodes[message.node_id] = node
            if message.node_id == message.thread_id:
                self.root = node
                self.id = node.message.thread_id
        self.size = len(list(self.nodes.items()))
        # print('conv_size:', self.size)
        self.participants = list(set(self.participants))
        if self.root is not None:
            # print("initializing conv: ", self.id)
            self.init_conversation_tree()


    def init_conversation_tree(self):
        '''initialize conversation tree'''
        queue = [self.root]
        # print("initing: ",self.root.message.node_id)
        while queue:
            current_node = queue.pop(0)
            # print("current_node_id: ", current_node.message.node_id )
            for node_id, node in self.nodes.items():
                # print("checking:", node.message.parent_id )
                if node.message.parent_id == current_node.message.node_id:
                    # print("found_child:", node.message.parent_id)
                    current_node.add_child(node)
                    queue.append(node)

    def iter_conversation(self) -> List[ConversationNode]:
        nodes_to_visit = [(0, self.root)]
        print("root_children: ", [node.message.node_id for node in self.root.children])
        while nodes_to_visit:
            depth, node = nodes_to_visit.pop(0)
            yield depth, node
            for child in node.children:
                nodes_to_visit.append((depth + 1, child))



def get_subtree_messages(root):
    '''get all messages in the subtree rooted at root_id'''
    queue = [root]
    messages = []
    while queue:
        current_node = queue.pop(0)
        message = current_node.message
        if type(message) == Message:
            messages.append(message)
        else:
            print(type(current_node))
            print(type(message))
        queue.extend(current_node.children)
    return messages
