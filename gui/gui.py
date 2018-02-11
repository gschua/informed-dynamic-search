# resizing: stackoverflow.com/q/22835289
# highlight: stackoverflow.com/q/3781670

import tkinter as tk
import tkinter.scrolledtext as tkst

def test_insert():
    global count
    global maps
    count += 1
    if count >= len(maps):
        count = 0
    with open(maps[count], 'r') as f:
        return f.read()

def xxx(event, map_box):
    current_location = float(map_box.index(tk.INSERT))
    map_box.config(state=tk.NORMAL)
    map_box.delete('1.0', tk.END)
    map_box.insert(tk.INSERT, test_insert())
    map_box.config(state=tk.DISABLED)
    # return to previous location
    map_box.mark_set('insert', current_location)
    map_box.see('insert')

DEFAULTS = {
    'win_width': 900,
    'win_height': 500,
    'map_width': 850,
    'map_height': 400,
    'map_padding': 10,
    'icon_side': 50,
}

count = 0
maps = [
    r'.\results\test\d08c01t11h03_map.csv',
    r'.\results\test\d08c01t11h04_map.csv',
    r'.\results\test\d08c01t11h05_map.csv',
    r'.\results\test\d08c01t11h06_map.csv',
    r'.\results\test\d08c01t11h07_map.csv',
    r'.\results\test\d08c01t11h08_map.csv',
    r'.\results\test\d08c01t11h09_map.csv',
    r'.\results\test\d08c01t11h10_map.csv',
    r'.\results\test\d08c01t11h11_map.csv',
    r'.\results\test\d08c01t11h12_map.csv',
]


#class ResizingText(tkst.ScrolledText):
#class ResizingCanvas(tk.Canvas):
class MapText(tk.Text):

    def __init__(self, parent, **kwargs):
        self.height_padding = kwargs.get('height_padding', 0)
        self.width_padding = kwargs.get('width_padding', 0)
        tk.Text.__init__(self, parent, **kwargs)
        self.bind("<Configure>", self.on_resize)
        #self.height = self.winfo_reqheight() - (self.width_padding * 2)
        self.height = self.winfo_reqheight()
        #self.width = self.winfo_reqwidth() - (self.height_padding * 2)
        self.width = self.winfo_reqwidth()

    # deals with resizing of windows
    def on_resize(self, event):
        # determine the ratio of old width/height to new width/height
        wscale = float(event.width)/self.width
        hscale = float(event.height)/self.height
        #self.width = event.width - (self.width_padding * 2)
        self.width = event.width
        #self.height = event.height - (self.height_padding * 2)
        self.height = event.height
        # resize the canvas 
        self.config(width=self.width, height=self.height)

    def highlight_pattern(self, pattern, tag, start="1.0", end="end", regexp=False):
        '''Apply the given tag to all text that matches the given pattern
        If 'regexp' is set to True, pattern will be treated as a regular
        expression according to Tcl's regular expression syntax.
        '''

        start = self.index(start)
        end = self.index(end)
        self.mark_set("matchStart", start)
        self.mark_set("matchEnd", start)
        self.mark_set("searchLimit", end)
        count = tk.IntVar()
        while True:
            index = self.search(pattern, "matchEnd","searchLimit", count=count, regexp=regexp)
            if index == "" or count.get() == 0: # degenerate pattern which matches zero-length strings
                break
            self.mark_set("matchStart", index)
            self.mark_set("matchEnd", "%s+%sc" % (index, count.get()))
            self.tag_add(tag, "matchStart", "matchEnd")

def main():

    # main window
    root = tk.Tk()
    root.geometry(str(DEFAULTS['win_width']) + 'x' + str(DEFAULTS['win_height']))

    # map box
    #map_frame = tk.Frame(root)
    #map_frame.pack(fill=tk.BOTH, expand=tk.YES)
    map_box = MapText(
        #map_frame,
        root,
        width=DEFAULTS['map_width'],
        height=DEFAULTS['map_height'],
        padx=DEFAULTS['map_padding'],
        pady=DEFAULTS['map_padding'],
        wrap="none",
        font='Consolas 7',
        bg='#1B2737',
        fg='white',
    )
    #map_box = ResizingCanvas(map_frame)
    map_box.insert(tk.INSERT, test_insert())


    # map tag configuration
    map_box.tag_configure("start", foreground="red", font='Consolas 10 bold')
    map_box.tag_configure("goal", foreground="green", font='Consolas 10 bold')
    map_box.tag_configure("visited", foreground="pink")

    # apply the tag "red" 
    map_box.highlight_pattern("o", "start")
    map_box.highlight_pattern("x", "goal")
    #map_box.highlight_pattern("+", "visited")
    #map_box.highlight_pattern("#", "visited")

    map_box.bind("<Return>", lambda event, map_box=map_box: xxx(event, map_box))

    # map scroll bars
    map_scroll_vertical = tk.Scrollbar(map_box, orient=tk.VERTICAL)
    map_scroll_vertical.pack(side=tk.RIGHT, fill=tk.Y)
    map_scroll_vertical.config(command=map_box.yview)

    map_scroll_horizontal = tk.Scrollbar(map_box, orient=tk.HORIZONTAL)
    map_scroll_horizontal.pack(side=tk.BOTTOM, fill=tk.X)
    map_scroll_horizontal.config(command=map_box.xview)

    map_box.config(xscrollcommand=map_scroll_horizontal.set, yscrollcommand=map_scroll_vertical.set)

    map_box.pack(fill=tk.BOTH, expand=tk.YES)
    root.mainloop()

if __name__ == "__main__":
    main()