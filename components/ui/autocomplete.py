import tkinter as tk
from tkinter import ttk

class Autocomplete:
    """Autocomplete widget for Tkinter text editor"""
    
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.listbox = None
        self.current_word = ""
        self.suggestions = []
        self.popup_active = False
        
        # Keywords lists based on PQL grammar
        self.keywords = [
            'var', 'follows', 'query', 'load_data', 'cluster', 'find_associations', 
            'classify', 'data', 'fitted_to'
        ]
        
        self.distributions = [
            'Normal', 'LogNormal', 'Poisson', 'Bernoulli', 'EmpiricalDistribution', 
            'Gamma', 'Beta', 'Multinomial'
        ]
        
        self.operators = ['and', 'or', 'not']
        
        # Define snippets with placeholders
        self.snippets = {
            'load_data': 'load_data("${filename.csv}", name: ${dataset_name});',
            'var': 'var ${variable_name} follows ${distribution};',
            'query P': 'query P(${condition});',
            'query E': 'query E(${variable});',
            'cluster': 'cluster(${dataset}, dimensions: [${dim1}, ${dim2}], k: ${3});',
            'find_associations': 'find_associations(${dataset}, min_support: ${0.3}, min_confidence: ${0.8});',
            'classify': 'classify(${dataset}, target: ${column_name}, features: [${feature1}, ${feature2}]);'
        }
        
        # Bind events
        self.text_widget.bind("<KeyRelease>", self.on_key_release)
        self.text_widget.bind("<Tab>", self.on_tab)
        self.text_widget.bind("<Down>", self.on_down)
        self.text_widget.bind("<Up>", self.on_up)
        self.text_widget.bind("<Return>", self.on_return)
        self.text_widget.bind("<Escape>", self.hide_popup)
        self.text_widget.bind("<Control-space>", self.force_autocomplete)
    
    def get_current_word(self):
        """Get the word under cursor for autocompletion"""
        cursor_pos = self.text_widget.index(tk.INSERT)
        line, col = map(int, cursor_pos.split("."))
        
        # Get current line text up to cursor
        line_text = self.text_widget.get(f"{line}.0", cursor_pos)
        
        # Find the word start
        word_start = col
        for i in range(col-1, -1, -1):
            if i < 0 or line_text[i].isspace() or line_text[i] in "();,{}[]":
                word_start = i + 1
                break
        
        # Extract the current word
        current_word = line_text[word_start:col]
        
        # Also get context before current word for better suggestions
        context_before = line_text[:word_start].strip()
        
        return current_word, context_before, f"{line}.{word_start}"
    
    def get_suggestions(self, word, context_before):
        """Get autocomplete suggestions based on current word and context"""
        suggestions = []
        
        # Different suggestions based on context
        if context_before.endswith("follows"):
            # After "follows", suggest distributions
            for dist in self.distributions:
                if dist.lower().startswith(word.lower()):
                    suggestions.append(dist)
        elif context_before.endswith("query"):
            # After "query", suggest query types
            suggestions = [s for s in ["P", "E", "correlation", "outliers"] if s.startswith(word)]
        else:
            # Default suggestions
            # Add keywords
            for kw in self.keywords:
                if kw.startswith(word) and len(word) > 0:
                    suggestions.append(kw)
            
            # Add distributions
            for dist in self.distributions:
                if dist.lower().startswith(word.lower()) and len(word) > 0:
                    suggestions.append(dist)
            
            # Add operators
            for op in self.operators:
                if op.startswith(word) and len(word) > 0:
                    suggestions.append(op)
            
            # Add snippets for special cases
            if "query".startswith(word) and len(word) > 0:
                suggestions.append("query P")
                suggestions.append("query E")
        
        return suggestions
    
    def show_popup(self, suggestions, word_start_pos):
        """Show autocomplete popup"""
        if not suggestions:
            self.hide_popup()
            return
        
        self.suggestions = suggestions
        
        # Calculate position for the popup
        try:
            x, y, width, height = self.text_widget.bbox(tk.INSERT)
            
            # If bbox returns None, use a different approach
            if x is None:
                cursor_pos = self.text_widget.index(tk.INSERT)
                x = self.text_widget.winfo_width() // 2
                y = int(cursor_pos.split('.')[0]) * 20  # Approximate height of a line
        except (TypeError, ValueError):
            # Fallback values
            x, y = 50, 50
        
        # Create a toplevel window if it doesn't exist
        if self.listbox is None:
            self.popup = tk.Toplevel(self.text_widget)
            self.popup.wm_overrideredirect(True)  # Remove window decorations
            self.popup.attributes("-topmost", True)  # Keep on top
            
            # Make sure it's visible
            self.popup.lift()
            
            # Color scheme that stands out
            self.listbox = tk.Listbox(
                self.popup,
                width=30,
                height=min(10, len(suggestions)),
                selectbackground="#4a6984",
                selectforeground="white",
                background="#1e1e1e",
                foreground="#d4d4d4",
                selectmode=tk.SINGLE
            )
            self.listbox.pack(fill=tk.BOTH, expand=True)
        else:
            self.listbox.delete(0, tk.END)
            self.listbox.config(height=min(10, len(suggestions)))
        
        # Populate listbox
        for i, item in enumerate(suggestions):
            self.listbox.insert(tk.END, item)
        
        # Select first item
        if suggestions:
            self.listbox.selection_set(0)
            self.listbox.activate(0)
        
        # Position the popup - make sure it's visible on screen
        try:
            offset_y = height + 2  # Position below the cursor with a small gap
            self.popup.geometry(f"+{x + self.text_widget.winfo_rootx()}+{y + offset_y + self.text_widget.winfo_rooty()}")
            
            # Make sure the popup is displayed
            self.popup.update_idletasks()
            self.popup.deiconify()
            self.popup_active = True
        except Exception as e:
            # If there's any error in positioning, use a safe fallback
            print(f"Error showing popup: {e}")
            self.popup.geometry("+50+50")
            self.popup.deiconify()
            self.popup_active = True
    
    def hide_popup(self, event=None):
        """Hide autocomplete popup"""
        if self.popup_active and self.listbox:
            self.popup.withdraw()
            self.popup_active = False
        return "break"
    
    def on_key_release(self, event):
        """Handle key release event"""
        # Ignore special keys
        if event.keysym in ("Up", "Down", "Return", "Tab", "Escape", "space"):
            return
        
        # Get current word
        current_word, context_before, word_start_pos = self.get_current_word()
        self.current_word = current_word
        
        # Only show popup if word is non-empty
        if len(current_word) > 0:
            suggestions = self.get_suggestions(current_word, context_before)
            if suggestions:
                self.show_popup(suggestions, word_start_pos)
            else:
                self.hide_popup()
        else:
            self.hide_popup()
    
    def force_autocomplete(self, event):
        """Force show autocomplete on Ctrl+Space"""
        current_word, context_before, word_start_pos = self.get_current_word()
        self.current_word = current_word
        
        suggestions = self.get_suggestions(current_word, context_before)
        self.show_popup(suggestions, word_start_pos)
        return "break"
    
    def on_tab(self, event):
        """Handle Tab key to accept suggestion"""
        if self.popup_active:
            return self.on_return(event)
        return None  # Allow default Tab behavior
    
    def on_down(self, event):
        """Handle Down arrow key"""
        if self.popup_active:
            # Move selection down
            current = self.listbox.curselection()
            if current:
                idx = int(current[0])
                if idx < len(self.suggestions) - 1:
                    self.listbox.selection_clear(0, tk.END)
                    self.listbox.selection_set(idx + 1)
                    self.listbox.activate(idx + 1)
                    self.listbox.see(idx + 1)
            return "break"
        return None
    
    def on_up(self, event):
        """Handle Up arrow key"""
        if self.popup_active:
            # Move selection up
            current = self.listbox.curselection()
            if current:
                idx = int(current[0])
                if idx > 0:
                    self.listbox.selection_clear(0, tk.END)
                    self.listbox.selection_set(idx - 1)
                    self.listbox.activate(idx - 1)
                    self.listbox.see(idx - 1)
            return "break"
        return None
    
    def on_return(self, event):
        """Handle Return key to accept suggestion"""
        if self.popup_active:
            current = self.listbox.curselection()
            if current:
                selected = self.suggestions[int(current[0])]
                self.insert_completion(selected)
            self.hide_popup()
            return "break"
        return None
    
    def insert_completion(self, selected):
        """Insert the selected completion"""
        cursor_pos = self.text_widget.index(tk.INSERT)
        line, col = map(int, cursor_pos.split("."))
        
        # Get current line text up to cursor
        line_text = self.text_widget.get(f"{line}.0", cursor_pos)
        
        # Find the word start
        word_start = col
        for i in range(col-1, -1, -1):
            if i < 0 or line_text[i].isspace() or line_text[i] in "();,{}[]":
                word_start = i + 1
                break
        
        # Delete current word
        self.text_widget.delete(f"{line}.{word_start}", cursor_pos)
        
        # Insert selected text or snippet
        if selected in self.snippets:
            # Insert snippet and handle placeholders
            self.insert_snippet(self.snippets[selected])
        else:
            # Just insert the completion
            self.text_widget.insert(f"{line}.{word_start}", selected)
    
    def insert_snippet(self, snippet_text):
        """Insert a snippet with placeholders"""
        # Find placeholders like ${placeholder}
        import re
        placeholders = re.finditer(r'\${([^}]*)}', snippet_text)
        
        # Replace placeholders with their text
        processed_text = re.sub(r'\${([^}]*)}', r'\1', snippet_text)
        
        # Insert the processed text
        cursor_pos = self.text_widget.index(tk.INSERT)
        self.text_widget.insert(cursor_pos, processed_text)
        
        # Find the first placeholder position to place cursor there
        first_placeholder = re.search(r'\${([^}]*)}', snippet_text)
        if first_placeholder:
            # Calculate cursor position after first placeholder
            start_idx = first_placeholder.start()
            placeholder_length = len(first_placeholder.group(1))
            
            # Get line and column of insertion point
            line, col = map(int, cursor_pos.split("."))
            
            # Set insertion point and select placeholder
            placeholder_start = f"{line}.{col + start_idx}"
            placeholder_end = f"{line}.{col + start_idx + placeholder_length}"
            
            self.text_widget.mark_set(tk.INSERT, placeholder_start)
            self.text_widget.see(tk.INSERT)
            self.text_widget.tag_add(tk.SEL, placeholder_start, placeholder_end)