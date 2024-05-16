import tkinter as tk
import tkinter.simpledialog
from tkinter import messagebox
import math
import random
import sys 
import ast

class Node:
    def __init__(self, name="None", id=-1, location=(-1,-1), text_id=-1):
        self.name = name
        self.id = id
        self.location = location
        self.text_id = text_id
        
class Edge:
    def __init__(self, name=("None", "None"), id=-1, start_node=Node(), end_node=Node()):
        self.name = name
        self.id = id
        self.start_node = start_node
        self.end_node = end_node

    

class NodeList():
    def __init__(self, in_nodes:list[Node]=[]):
        self.node_list = in_nodes
    
    def __add__(self, new: Node):
        self.node_list.append(new)
        
    def __iter__(self):
        return self.node_list.__iter__()
    
    def __len__(self):
        return len(self. node_list)
        
    def append(self, new: Node):
        self.node_list.append(new)
        
    def pop(self, ind=-1):
        self.node_list.pop(ind)

    def contains_name(self, query):
        for i in self.node_list:
            if i.name == query:
                return True
        return False
    
    def contains_id(self, query):
        for i in self.node_list:
            if i.id == query:
                return True
        return False

    def contains_location(self, query):
        for i in self.node_list:
            if i.location == query:
                return True
        return False
    
    def get_name(self, query):
        for i in self.node_list:
            if i.name == query:
                return i
        return None

    def get_id(self, query):
        for i in self.node_list:
            if i.id == query:
                return i
        return None
    
    def get_location(self, query):
        for i in self.node_list:
            if i.location == query:
                return i
        return None
    
    def dump(self):
        return [x.name for x in self.node_list]
    
    def remove(self, target):
        self.node_list = [x for x in self.node_list if x.name != target]
    
class EdgeList():
    def __init__(self, in_edges:list[Edge]=[]):
        self.edge_list = in_edges
    
    def __add__(self, new: Edge):
        self.edge_list.append(new)
        
    def __iter__(self):
        return self.edge_list.__iter__()
    
    def __len__(self):
        return len(self.edge_list)  
    
    def append(self, new: Edge):
        self.edge_list.append(new)
        
    def pop(self, ind=-1):
        self.edge_list.pop(ind)
        
    def contains_name(self, query):
        for i in self.edge_list:
            if i.name == query:
                return True
        return False
    
    def remove(self, target):
        self.edge_list = [x for x in self.edge_list if x.name != target]
    
    def dump(self):
        return [x.name for x in self.edge_list]
    
        

class Dag_Drawer:
    def __init__(self, in_nodes=[], in_edges=[], size=2000, term=False):
        root = tk.Tk()
        root.title("DAG Drawer")
        root.bind("<Command-z>", self.undo)
        root.bind("<Command-Z>", self.redo)
        root.bind("<Shift-Button-1>", self.move_node)

        root.bind("t", self.toggle)
        root.bind("p", self.dump)
        root.bind("m", self.move_node)

        
        
        root.bind("<Escape>", self.clear)
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        size = min(size, screen_width, screen_height)
        self.size = size
        self.canvas = tk.Canvas(root, width=size, height=size, bg='black')
        self.canvas.pack()
        
        self.num_edges = tk.StringVar()
        self.num_edges.set("Edges: 0")
        self.num_nodes = tk.StringVar()
        self.num_nodes.set("Nodes: 0")
        
        self.edge_count_id = self.canvas.create_text(20, 20, anchor=tk.NW, text=self.num_edges.get(), font=('Arial', 16), fill='white')
        self.node_count_id = self.canvas.create_text(20, 40, anchor=tk.NW, text=self.num_nodes.get(), font=('Arial', 16), fill='white')
        
        
        self.moving = None
        
        
        self.canvas.bind("<Button-1>", self.handle_click)
        self.not_active = "Edges"
        button_font = ('Arial', 16, 'bold')
        self.button = tk.Button(self.canvas, text=f"Switch to Adding {self.not_active}", command=self.toggle, highlightbackground='black', font=button_font,
                          padx=10, pady=10)
        button_window = self.canvas.create_window(size - size / 20, size / 20, anchor=tk.NE, window=self.button)
        
        print_button = tk.Button(self.canvas, text=f"{'Return' if term else 'Print'} Nodes and Edges", command=self.dump, highlightbackground='black', font=button_font,
                          padx=10, pady=10)
        button_window = self.canvas.create_window(size - size / 20, size - 2 * (size / 20), anchor=tk.SE, window=print_button)
        
        
        self.undo_stack = []
        self.redo_stack = []
        self.term = term
        
        
        #important node lists
        # self.nodes = []
        # self.node_ids = []
        # self.node_locations = []
        
        self.node_data: NodeList = NodeList()
        
        # self.edges = []
        # self.curr_edge = []
        self.half_edge = None
        self.edge_data: EdgeList = EdgeList()
        
        self.curr_edge_location = None
        
        self.import_nodes_and_edges(in_nodes, in_edges)
        
        root.mainloop()


    def update_text(self):
        self.canvas.itemconfig(self.edge_count_id, text=f"Edges: {len(self.edge_data)}")
        self.canvas.itemconfig(self.node_count_id, text=f"Nodes: {len(self.node_data)}")


    def import_nodes_and_edges(self, in_nodes, in_edges):
        max_rand = int(self.size * 0.875)
        min_rand = int(self.size * 0.125)

        
        for i in in_nodes:
            self.draw_node_help(random.randint(min_rand, max_rand), random.randint(min_rand, max_rand), i)
            
        for i in in_edges:
            s_node = self.node_data.get_name(i[0])
            e_node = self.node_data.get_name(i[1])
            
            self.draw_edge_helper(s_node, e_node)

    
    def dump(self, event=None):
 
        print("nodes =", self.node_data.dump())
        print("edges =", self.edge_data.dump())
                   
        if self.term:
            exit(0)
        
        
    def toggle(self, event=None):
        # note toggling clears undo stack intentionally
        # dont want to draw an edge and then undo
        self.not_active = "Nodes" if self.not_active == "Edges" else "Edges"
        self.button.config(text=f"Switch to Adding {self.not_active}")
        # self.undo_stack = []

    def clear(self, event):
        if self.not_active == "Nodes":
            if self.half_edge is not None:
                self.canvas.itemconfig(self.half_edge.id, outline='white')
            self.half_edge = None
        
        if self.moving is not None:
            self.canvas.itemconfig(self.moving.id, outline='white')
            self.moving = None
            
            # self.curr_edge_location = None
            # [self.canvas.itemconfig(x.id, outline='white') for x in self.node_data]
            
    # implement undo into move
    def move_node(self, event):
            if self.moving is None:
                my_node = self.find_closest_node(event)
                if my_node is not None:
                    self.half_edge = None
                    
                    self.moving = my_node
                    self.canvas.itemconfig(self.moving.id, outline='limegreen')

                else:
                    messagebox.showinfo("Node", "No node found")
                

            else:
                self.canvas.delete(self.moving.id)
                self.canvas.delete(self.moving.text_id)
                self.node_data.remove(self.moving.name)
                x = event.x
                y = event.y
                
                my_text = self.canvas.create_text(x, y-20, text=self.moving.name, fill='white')
                my_circle = self.canvas.create_oval(x-10, y-10, x+10, y+10, outline='white', width=2)
                
                new_node = Node(self.moving.name, my_circle, (x, y), my_text)
                self.node_data.append(new_node)
                self.moving = None
                
                for edge in self.edge_data:
                    start = edge.start_node
                    end = edge.end_node
                    
                    
                    if start.name == new_node.name:
                        
                        self.canvas.delete(edge.id)
                        self.edge_data.remove(edge.name)
                        self.draw_edge_helper(new_node, end)

                        
                    if end.name == new_node.name:
                        
                        self.canvas.delete(edge.id)
                        self.edge_data.remove(edge.name)
                        self.draw_edge_helper(start, new_node)
        
            
            
        # do something when mouse is released
        
    
    
    def draw_node(self, event):
        x, y = event.x, event.y
        self.draw_node_help(x, y)
        
    
    def draw_node_help(self, x, y, text=None):
        # x, y = event.x, event.y
        if text is None:
            text = tkinter.simpledialog.askstring("Text Input", "Enter text:")
        if text:
            
            if (not self.node_data.contains_name(text)):
                # self.node_data[text] = 
                # self.nodes.append(text)
                # self.node_locations.append((x,y))
                
                my_text = self.canvas.create_text(x, y-20, text=text, fill='white')
                my_circle = self.canvas.create_oval(x-10, y-10, x+10, y+10, outline='white', width=2)
                
                new_node = Node(text, my_circle, (x,y), my_text)
                self.node_data.append(new_node)
                self.undo_stack.append(new_node)
                self.update_text()
            
            else:
                messagebox.showinfo("Repeat", "Error: this node already exists")
            
            
    # TODO: make undo not only for deleting
    def undo(self, event):
        if len(self.undo_stack) > 0:
            
            
            item: Node | Edge = self.undo_stack.pop()
            if type(item) == Node:
                self.redo_stack.append(item)
                self.canvas.delete(item.id)
                self.canvas.delete(item.text_id)
                self.node_data.pop()
            
            else:
                self.redo_stack.append(item)
                self.canvas.delete(item.id)
                self.edge_data.pop()
            
            self.update_text()
            
                
        else:
            messagebox.showinfo("Undo", "Nothing to undo")
            
    def redo(self, event):
        if len(self.redo_stack) > 0:
                        
            
            item = self.redo_stack.pop()
            if type(item) == Node:
                
                x = item.location[0]
                y = item.location[1]
                
                self.draw_node_help(x, y, item.name)
                
            
            else:
                start = item.start_node
                end = item.end_node
                self.draw_edge_helper(start, end)

            self.update_text()
            
        else:
            messagebox.showinfo("Redo", "Nothing to redo")
            
    def find_closest_node(self, event, threshhold=50):
        # print("here")
        x, y = event.x, event.y
        best_node = None
        best = threshhold + 1
        # for i, n in enumerate(self.node_locations):
        for i in self.node_data:
            nx, ny = i.location
            distance_from_click = ((nx - x)**2 + (ny - y)**2)**0.5
            if distance_from_click < threshhold:
                if distance_from_click < best:
                    best = distance_from_click
                    # self.curr_edge.append(i)
                    best_node = i
                    # add edge is from nodes[best_index]
        return best_node
    

    def line_adjust(self, x1, y1, x2, y2):
        # Calculate the distance between the centers of the circles
        dist_centers = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Calculate the unit vector along the line connecting the centers
        if dist_centers == 0:
            return x1 + 10, y1, x2 - 10, y2  # If the circles have the same center, return points on x-axis
        
        ux = (x2 - x1) / dist_centers
        uy = (y2 - y1) / dist_centers
        
        # Calculate the points on the line segment closest to the circles' perimeters
        x1_new = x1 + 10 * ux
        y1_new = y1 + 10 * uy
        x2_new = x2 - 10 * ux
        y2_new = y2 - 10 * uy
        
        return x1_new, y1_new, x2_new, y2_new


    def draw_edge(self, event):
        if self.half_edge is not None:
            self.half_edge.location
            my_end = self.find_closest_node(event)
            if my_end is None:
                messagebox.showinfo("Node", "No node found")
            elif self.half_edge.name == my_end.name:
                messagebox.showinfo("Node", "Can't add connect node to itself")
                
            
            else:
                self.draw_edge_helper(self.half_edge, my_end)
        
        else:
            self.half_edge = self.find_closest_node(event)
            if self.half_edge is not None:
                self.canvas.itemconfig(self.half_edge.id, outline='red')
            
            if self.half_edge is None:
                messagebox.showinfo("Node", "No node found")
                self.curr_edge_location = None
            
    
    def draw_edge_helper(self, start, end): #x1, y1, x2, y2):

        if end.location == (-1, -1):
            messagebox.showinfo("Node", "No node found")
            return
        # TODO: should do math to make arrows appear on perimeter
        x1, y1 = start.location
        x2, y2 = end.location
        
        ax1, ay1, ax2, ay2 = self.line_adjust(x1, y1, x2, y2)
        my_name = (start.name, end.name)
        
        #TODO undo / redo edges
        self.canvas.itemconfig(start.id, outline='white')
        self.canvas.itemconfig(end.id, outline='white')
        
        self.half_edge = None
        if (self.edge_data.contains_name(my_name)):
            
            messagebox.showinfo("Repeat", "Error: this edge already exists")
            # return
        else:
            my_line = self.canvas.create_line(ax1, ay1, ax2, ay2, arrow=tk.LAST) # maybe .FIRST
            my_edge = Edge(my_name, my_line, start, end)
            self.edge_data.append(my_edge)
            self.undo_stack.append(my_edge)
            self.update_text()
            
        
        
        # self.edges.append(tuple(self.curr_edge))
        # self.curr_edge = []

        
            
    
    def handle_click(self, event):
        if not event.state & 0x1:
            if self.moving is None:
                if self.not_active == "Edges":
                    # drawing nodes
                    self.draw_node(event)
                else:
                    self.draw_edge(event)
            else:
                self.move_node(event)
            
        # print(self.nodes, self.edges)
        
            
        
        
#TODO: work on better existing imports

#TODO: make things moveable
#TODO: have a edge and node counter
#TODO: could detect cycles

def main():
    nodes = []
    edges = []
    drawer = Dag_Drawer(nodes, edges, term=True)
    
if __name__ == "__main__":
    main()
