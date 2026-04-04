import os
import sys

# Set Tcl/Tk paths for Windows pyenv/virtualenv environments
# This must be done BEFORE importing tkinter or customtkinter
tcl_path = r"C:\Users\My PC\.pyenv\pyenv-win\versions\3.13.0\tcl\tcl8.6"
tk_path = r"C:\Users\My PC\.pyenv\pyenv-win\versions\3.13.0\tcl\tk8.6"

if os.path.exists(tcl_path) and os.path.exists(tk_path):
    os.environ["TCL_LIBRARY"] = tcl_path
    os.environ["TK_LIBRARY"] = tk_path
else:
    print(f"Warning: Tcl/Tk paths not found:\n{tcl_path}\n{tk_path}")

import customtkinter as ctk
import threading
import webbrowser
from PIL import Image, ImageTk

# Import core logic
from core import (
    SpotifyClient, 
    IntentParser, 
    ChatBot, 
    LocalMusicLibrary, 
    get_sp_oauth, 
    GROQ_API_KEY
)

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        # Since this is in src/app.py, the project root is one dir up
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme(get_resource_path("assets/themes/pink.json"))

class ChimEkenApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("Chim-eTunes 🐔🎶")
        self.geometry("1000x600")

        # Set window icon for title bar and taskbar
        try:
            self.iconbitmap(get_resource_path("assets/images/Chim-eTunes Logo.ico"))
        except:
            pass  # Fallback if icon not found
        
        # Make window resizable with limits to maintain UI integrity
        self.resizable(True, True)
        
        # Size limits: minimum 800x600 
        self.minsize(800, 600)
        
        # Automatically detect screen size and set as maximum
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.maxsize(screen_width, screen_height)
        
        # Grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # State
        self.local_lib = None
        self.sp_client = None
        self.bot = None
        self.history = []
        self.session_data = {} # For storing last_recs etc.

        # --- UI Components ---
        self._create_sidebar()
        self._create_chat_area()
        self._create_input_area()
        self._update_chat_colors()

        # --- Initialization ---
        self.after(100, self._init_backend)

    def _create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar.grid_rowconfigure(6, weight=1)

        # Title
        try:
            logo_image = ctk.CTkImage(
                light_image=Image.open(get_resource_path("assets/images/Chim-eTunes Logo.png")).resize((130, 120)),
                dark_image=Image.open(get_resource_path("assets/images/Chim-eTunes Logo.png")).resize((130, 120)),
                size=(150, 120) 
            )
            self.logo_label = ctk.CTkLabel(self.sidebar, image=logo_image, text="")
            self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        except Exception as e:
            print(f"Logo image error: {e}")
            # Fallback to text if image not found
            self.logo_label = ctk.CTkLabel(self.sidebar, text="Chim-eTunes", font=ctk.CTkFont(size=24, weight="bold"), text_color=["#FF6B6B", "#B74177"])
            self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Chim-eTunes Text Label
        self.logo_text_label = ctk.CTkLabel(
            self.sidebar,
            text="Chim-eTunes",
            font=ctk.CTkFont(family="Trebuchet MS", size=20, weight="bold"),
            text_color=["#C71585", "#FF69B4"]
        )
        self.logo_text_label.grid(row=1, column=0, padx=20, pady=(0, 15))

        # Status
        self.status_label = ctk.CTkLabel(self.sidebar, text="Initializing...")
        self.status_label.grid(row=2, column=0, padx=20, pady=10)

        # Spotify Login Button
        self.login_btn = ctk.CTkButton(self.sidebar, text="Log in with Spotify", command=self._login_spotify)
        self.login_btn.grid(row=3, column=0, padx=20, pady=10)
        
        # Spotify Logout Button (initially hidden)
        self.logout_btn = ctk.CTkButton(self.sidebar, text="Logout from Spotify", command=self._logout_spotify)
        self.logout_btn.grid(row=4, column=0, padx=20, pady=10)
        self.logout_btn.grid_remove()  # Hide initially

        # Dark/Light Mode Switch
        self.mode_switch = ctk.CTkSwitch(self.sidebar, text="Dark Mode", command=self._toggle_mode)
        self.mode_switch.grid(row=5, column=0, padx=20, pady=10)
        self.mode_switch.select()  # Start in dark mode
        
        # Clear Chat
        self.clear_btn = ctk.CTkButton(self.sidebar, text="Clear Chat", command=self._clear_chat, border_width=2)
        self.clear_btn.grid(row=6, column=0, padx=20, pady=10)

    def _create_chat_area(self):
        self.chat_box = ctk.CTkTextbox(self, state="disabled", wrap="word", font=ctk.CTkFont(size=14))
        self.chat_box.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        # Configure tags for colors (accessing underlying tkinter widget)
        # Using brighter colors for dark mode visibility
        self.chat_box._textbox.tag_config("user") 
        self.chat_box._textbox.tag_config("bot") 
        self.chat_box._textbox.tag_config("system")
        self.chat_box._textbox.tag_config("sender_header", font=ctk.CTkFont(size=14, weight="bold"))

    def _create_input_area(self):
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.grid(row=1, column=1, padx=20, pady=(0, 20), sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)



        def auto_resize(event):
            # Count lines (split by newline char)
            text = self.entry.get("1.0", "end-1c")
            num_lines = text.count('\n') + 1
            
            # Calculate new height (approx 20px per line + padding)
            new_height = min(max(50, num_lines * 20 + 10), 150) # Min 50, Max 150
            
            # Only update if changed to avoid flickering
            if self.entry.cget("height") != new_height:
                self.entry.configure(height=new_height)

        # Typing Indicator
        self.typing_label = ctk.CTkLabel(self.input_frame, text="Chim-eken is typing... 🐔", font=ctk.CTkFont(size=12, slant="italic"), text_color="gray")
        self.typing_label.grid(row=0, column=0, padx=20, pady=(0, 5), sticky="w")
        self.typing_label.grid_remove()

        self.entry = ctk.CTkTextbox(self.input_frame, height=50, wrap="word", font=ctk.CTkFont(size=14))
        self.entry.grid(row=1, column=0, padx=(20, 10), pady=10, sticky="ew")

        self.entry.bind("<Return>", self._on_enter)
        self.entry.bind("<KeyRelease>", auto_resize)

        self.send_btn = ctk.CTkButton(self.input_frame, text="Send", width=100, command=self._send_message)
        self.send_btn.grid(row=1, column=1, padx=(0, 20), pady=10, sticky="s")

    def _init_backend(self):
        # Load Local Library in background
        threading.Thread(target=self._load_local_lib, daemon=True).start()
        
        # Check for cached Spotify token
        self._check_spotify_token()
        
        # Init Intent Parser
        self.parser = IntentParser(GROQ_API_KEY)
        
        # Init Bot (initially without Spotify or Local, will update)
        self.bot = ChatBot(self.sp_client, self.parser, self.local_lib)

        # Show loading message immediately
        self._add_message("Chim-eken", "Hello! I'm Chim-eken. 🐔\nLoading music library... this will take a few seconds.")

    def _load_local_lib(self):
        self.local_lib = LocalMusicLibrary()
        if self.bot:
            self.bot.local = self.local_lib
        
        # Update UI
        self.status_label.configure(text="Local Library Ready ✅", text_color="green")

        # Update welcome message
        self._add_message("Chim-eken", "✅ Library loaded! I'm ready to help you discover music. Try asking for songs by genre, mood, or artist!")

    def _check_spotify_token(self):
        try:
            sp_oauth = get_sp_oauth()
            token_info = sp_oauth.get_cached_token()
            if token_info:
                self.sp_client = SpotifyClient(token_info)
                if self.bot: self.bot.sp = self.sp_client
                
                # Get user name and update UI
                try:
                    me = self.sp_client.current_user()
                    name = me.get('display_name') or me.get('id')
                    self.login_btn.configure(text=f"👤 {name}", state="disabled")
                    self.logout_btn.grid()  # Show logout button
                except:
                    pass
        except Exception as e:
            print(f"Token check failed: {e}")

    def _login_spotify(self):
        def login_thread():
            try:
                sp_oauth = get_sp_oauth()
                # This will open browser and wait for callback
                token_str = sp_oauth.get_access_token(as_dict=False) 
                if token_str:
                    token_info = {"access_token": token_str}
                    self.sp_client = SpotifyClient(token_info)
                    if self.bot: self.bot.sp = self.sp_client
                    
                    name = "User" # Default
                    try:
                        me = self.sp_client.current_user()
                        name = me.get('display_name') or me.get('id')
                    except:
                        pass
                    # Update UI on main thread
                    self.after(0, lambda: self.login_btn.configure(text=f"👤 {name}", state="disabled"))
                    self.after(0, lambda: self.logout_btn.grid())  # Show logout button
                    self.after(0, lambda: self._add_message("System", "Spotify login successful! You can now manage playlists."))
                    self.after(0, lambda: self.deiconify())
                    self.after(0, lambda: self.lift())
                    self.after(0, lambda: self.focus_force())
                    
            except Exception as e:
                print(f"Login failed: {e}")
                error_msg = f"Login failed: {e}"
                self.after(0, lambda: self._add_message("System", error_msg))

        threading.Thread(target=login_thread, daemon=True).start()

    def _logout_spotify(self):
        """Logout from Spotify by clearing cached token"""
        try:
            # Clear the cached token file
            import os
            cache_path = ".cache"
            if os.path.exists(cache_path):
                os.remove(cache_path)
            
            # Reset client and bot
            self.sp_client = None
            if self.bot:
                self.bot.sp = None
            
            # Update UI
            self.login_btn.configure(text="Log in with Spotify", state="normal")
            self.logout_btn.grid_remove()  # Hide logout button
            self._add_message("System", "Logged out from Spotify. Playlist features are now disabled.")
        except Exception as e:
            print(f"Logout failed: {e}")
            self._add_message("System", f"Logout failed: {e}")

    def _toggle_mode(self):
        """Toggle between dark and light mode"""
        if self.mode_switch.get():
            ctk.set_appearance_mode("Dark")
            self.mode_switch.configure(text="Dark Mode")
        else:
            ctk.set_appearance_mode("Light")
            self.mode_switch.configure(text="Light Mode")
        self._update_chat_colors()

    def _update_chat_colors(self):
        mode = ctk.get_appearance_mode()
        if mode == "Dark" or mode == "System":
            self.chat_box._textbox.tag_config("user", foreground="#FFD700")
            self.chat_box._textbox.tag_config("bot", foreground="#FF6B6B")
            self.chat_box._textbox.tag_config("system", foreground="#A0AEC0")
        else:
            self.chat_box._textbox.tag_config("user", foreground="#800080")  
            self.chat_box._textbox.tag_config("bot", foreground="#C71585")  
            self.chat_box._textbox.tag_config("system", foreground="#4A5568")

    def _on_enter(self, event):
        if event.state & 0x0001: 
            return # Let default behavior happen (newline)
            
        # Otherwise send message and prevent default newline
        self._send_message()
        return "break"

    def _send_message(self):
        text = self.entry.get("1.0", "end-1c").strip()
        if not text: return

        self.entry.delete("1.0", "end")
        self._add_message("You", text)

        # Disable input while processing
        self.entry.configure(state="disabled")
        self.send_btn.configure(state="disabled")

        # Show typing indicator
        self.typing_label.grid()
        
        # Process in background
        threading.Thread(target=self._process_message, args=(text,), daemon=True).start()

    def _process_message(self, text):
        if not self.bot: return
        
        # Call bot logic
        reply, items, new_history = self.bot.handle_message(text, self.history, self.session_data)
        
        # Update history
        self.history = new_history
        
        # Update UI
        self.after(0, lambda: self._add_message("Chim-eken", reply))
        
        if items:
            # Check if items are image URLs
            if items and all(isinstance(item, str) and item.startswith('http') and 
                            ('scdn.co/image' in item or 'spotifycdn.com' in item or
                             any(ext in item.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif'])) 
                            for item in items):
                # These are images - display them
                for img_url in items:
                    self.after(0, lambda url=img_url: self._add_image(url))
            else:
                # Regular text items (song recommendations, etc.)
                formatted_items = "\n".join([f"• {it}" for it in items])
                self.after(0, lambda: self._add_message("Chim-eken", formatted_items))
        
        # Re-enable input
        def enable_input():
            self.typing_label.grid_remove()
            self.entry.configure(state="normal")
            self.send_btn.configure(state="normal")
            self.entry.focus()
        
        self.after(0, enable_input)

    def _add_message(self, sender, text):
        self.chat_box.configure(state="normal")
        
        tag = "system"
        if sender == "Chim-eken": tag = "bot"
        elif sender == "You": tag = "user"
        
        # Insert Sender Name
        self.chat_box.insert("end", f"{sender}:\n", (tag, "sender_header"))
        
        # Insert Message Body
        self._insert_text_with_links(text)
        self.chat_box.insert("end", "\n\n")
        
        self.chat_box.configure(state="disabled")
        self.chat_box.see("end")

    def _insert_text_with_links(self, text):
        """Insert text and make URLs clickable"""
        import re
        
        # Regex to find URLs
        url_pattern = r'(https?://[^\s]+)'
        parts = re.split(url_pattern, text)
        
        for part in parts:
            if re.match(url_pattern, part):
                # This is a URL - make it clickable
                start_index = self.chat_box.index("end-1c")
                self.chat_box.insert("end", part)
                end_index = self.chat_box.index("end-1c")
                
                # Create a unique tag for this link
                tag_name = f"link_{start_index}"
                self.chat_box._textbox.tag_add(tag_name, start_index, end_index)
                self.chat_box._textbox.tag_config(tag_name, foreground="#1E90FF", underline=True)
                
                # Bind click event
                self.chat_box._textbox.tag_bind(tag_name, "<Button-1>", 
                    lambda e, url=part: self._open_link(url))
                
                # Change cursor on hover
                self.chat_box._textbox.tag_bind(tag_name, "<Enter>", 
                    lambda e: self.chat_box._textbox.config(cursor="hand2"))
                self.chat_box._textbox.tag_bind(tag_name, "<Leave>", 
                    lambda e: self.chat_box._textbox.config(cursor=""))
            else:
                # Regular text
                self.chat_box.insert("end", part)
    
    def _open_link(self, url):
        """Open URL in default browser"""
        import webbrowser
        webbrowser.open(url)

    def _clear_chat(self):
        self.chat_box.configure(state="normal")
        self.chat_box.delete("0.0", "end")
        self.chat_box.configure(state="disabled")
        self.history = []

def _add_image(self, image_url):
    """Download and display image inline in chat"""
    try:
        import requests
        from io import BytesIO
        from PIL import ImageTk  # ← Use ImageTk instead of CTkImage
        
        # Download image
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        
        # Load image
        img_data = BytesIO(response.content)
        img = Image.open(img_data)
        
        # Resize if too large (max 300x300)
        max_size = (300, 300)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage (works with tkinter.Text)
        photo = ImageTk.PhotoImage(img)
        
        # Insert into chat
        self.chat_box.configure(state="normal")
        
        # Use the underlying tkinter Text widget
        self.chat_box._textbox.image_create("end", image=photo)
        self.chat_box.insert("end", "\n")
        
        # Keep reference to prevent garbage collection
        if not hasattr(self, '_images'):
            self._images = []
        self._images.append(photo)
        
        self.chat_box.configure(state="disabled")
        self.chat_box.see("end")
        
    except Exception as e:
        print(f"Failed to load image: {e}")
        self._add_message("Chim-eken", f"Image: {image_url}")

if __name__ == "__main__":
    app = ChimEkenApp()
    app.mainloop()
 