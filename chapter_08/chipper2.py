#  Chipper for selecting square images, GPT-4o, 09-Sep-2024
#  v2 tweaks by RTK, 15-Dec-2024

import wx
import os
import sys
import numpy as np
from PIL import Image

# Helper functions
def pil_to_wx(image):
    """Convert PIL image to wx.Image."""
    image = image.convert('RGB')  # Ensure the image is RGB (no alpha)
    width, height = image.size
    image_data = np.array(image)
    image_wx = wx.Image(width, height)
    image_wx.SetData(image_data.tobytes())
    return image_wx

def extract_square_chip(image_array, click_x, click_y):
    """Extract the largest square with click_x, click_y at the center."""
    height, width, _ = image_array.shape
    half_size = min(click_x, width - click_x, click_y, height - click_y)
    square_chip = image_array[click_y - half_size: click_y + half_size, click_x - half_size: click_x + half_size, :]
    return square_chip

class ImagePanel(wx.Panel):
    def __init__(self, parent, outdir):
        super().__init__(parent)
        self.bitmap = None
        self.image_path = None
        self.image_array = None

        self.outdir = outdir

        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_click)

    def load_image(self, image_path):
        """Load an image from a file."""
        self.image_path = image_path
        pil_image = Image.open(image_path).convert('RGB')  # Open and convert to RGB
        self.image_array = np.array(pil_image)
        self.update_bitmap()

    def update_bitmap(self):
        """Update the wx.Bitmap from the loaded image."""
        if self.image_array is not None:
            height, width, _ = self.image_array.shape
            wx_image = wx.Image(width, height)
            wx_image.SetData(self.image_array.tobytes())
            self.bitmap = wx.Bitmap(wx_image)
            self.Refresh()

    def on_paint(self, event):
        """Handle the paint event to draw the image."""
        if self.bitmap:
            dc = wx.PaintDC(self)
            w, h = self.GetSize()
            image_w, image_h = self.bitmap.GetSize()
            aspect_ratio = min(w / image_w, h / image_h)
            new_w = int(image_w * aspect_ratio)
            new_h = int(image_h * aspect_ratio)
            x_offset = (w - new_w) // 2
            y_offset = (h - new_h) // 2

            # Convert wx.Bitmap to wx.Image, scale it, and convert it back to wx.Bitmap
            wx_image = self.bitmap.ConvertToImage()
            scaled_image = wx_image.Scale(new_w, new_h, wx.IMAGE_QUALITY_HIGH)
            scaled_bitmap = wx.Bitmap(scaled_image)

            # Draw the scaled bitmap
            dc.DrawBitmap(scaled_bitmap, x_offset, y_offset, True)

    def on_click(self, event):
        """Handle mouse click event to extract and save a square chip."""
        if self.image_array is None:
            return
        x, y = event.GetPosition()
        # Find the scale factor to map click to the original image
        panel_w, panel_h = self.GetSize()
        image_h, image_w, _ = self.image_array.shape
        aspect_ratio = min(panel_w / image_w, panel_h / image_h)
        offset_x = (panel_w - int(image_w * aspect_ratio)) // 2
        offset_y = (panel_h - int(image_h * aspect_ratio)) // 2

        # Map the click to the original image coordinates
        img_x = int((x - offset_x) / aspect_ratio)
        img_y = int((y - offset_y) / aspect_ratio)

        if 0 <= img_x < image_w and 0 <= img_y < image_h:
            chip = extract_square_chip(self.image_array, img_x, img_y)
            self.save_chip(chip)

    def save_chip(self, chip):
        """Save the square chip to the chips directory."""
        if self.image_path is None:
            return
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        base_name = os.path.basename(self.image_path)
        name, ext = os.path.splitext(base_name)
        chip_name = f"{name}_chip.png"
        chip_path = os.path.join(self.outdir, chip_name)
        
        #  Save chips
        img = np.array(Image.fromarray(chip).resize((200,200), resample=Image.BILINEAR))
        k = 0
        for i in [-15,-10,-5,0,5,10,15]:
            for j in [-15,-10,-5,0,5,10,15]:
                im = Image.fromarray(img[(100+i-32):(100+i+32), (100+j-32):(100+j+32), :])
                chip_name = "%s_chip_%02d.png" % (name,k)
                chip_path = os.path.join(self.outdir, chip_name)
                im.save(chip_path)
                k += 1
        print("Saved:", chip_path)

class MainFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        
        self.last_directory = "~"
        self.panel = ImagePanel(self, sys.argv[1])

        self.create_menu()
        self.SetSize((900, 675))
        self.SetTitle('Chip Extractor 2')
        self.Centre()

    def create_menu(self):
        """Create the menu bar."""
        menubar = wx.MenuBar()
        file_menu = wx.Menu()

        open_item = file_menu.Append(wx.ID_OPEN, '&Open Image\tCtrl+O', 'Open an image')
        quit_item = file_menu.Append(wx.ID_EXIT, '&Quit\tCtrl+Q', 'Quit application')

        menubar.Append(file_menu, '&File')
        self.SetMenuBar(menubar)

        self.Bind(wx.EVT_MENU, self.on_open_image, open_item)
        self.Bind(wx.EVT_MENU, self.on_quit, quit_item)

    def on_open_image(self, event):
            """Handle the image open dialog."""
            
            # Use the last directory or default to the user's home directory
            default_dir = self.last_directory
            
            with wx.FileDialog(self, "Open Image file", 
                               wildcard="Image files (*.jpg;*.png;*.bmp)|*.jpg;*.png;*.bmp",
                               defaultDir=default_dir, 
                               style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
                
                if fileDialog.ShowModal() == wx.ID_CANCEL:
                    return  # User cancelled
                
                # Proceed to load the image
                image_path = fileDialog.GetPath()
                self.panel.load_image(image_path)

                #  Set the window title to the file name
                base_filename = os.path.basename(image_path)
                self.SetTitle(base_filename)

                # Update the last directory used
                self.last_directory = fileDialog.GetDirectory()

    def on_quit(self, event):
        """Handle quitting the application."""
        self.Close(True)

class MyApp(wx.App):
    def OnInit(self):
        frame = MainFrame(None)
        frame.Show(True)
        return True

if __name__ == '__main__':
    #  Command line options, RTK
    if (len(sys.argv) == 1):
        print()
        print("chipper2 <outdir>")
        print()
        print("  <outdir> - output chip directory")
        print()
        exit(0)

    app = MyApp(False)
    app.MainLoop()

