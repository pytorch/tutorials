Using Tutorial Data from Google Drive in Colab
==============================================

We've added a new feature to tutorials that allows users to open the
notebook associated with a tutorial in Google Colab. You may need to
copy data to your Google drive account to get the more complex tutorials
to work.

In this example, we'll demonstrate how to change the notebook in Colab
to work with the Chatbot Tutorial. To do this, you'll first need to be
logged into Google Drive. (For a full description of how to access data
in Colab, you can view their example notebook
`here <https://colab.research.google.com/notebooks/io.ipynb#scrollTo=XDg9OBaYqRMd>`__.)

To get started open the `Chatbot
Tutorial <https://pytorch.org/tutorials/beginner/chatbot_tutorial.html>`__
in your browser.

At the top of the page click **Run in Google Colab**.

The file will open in Colab.

If you select **Runtime**, and then **Run All**, you'll get an error as the
file can't be found.

To fix this, we'll copy the required file into our Google Drive account.

1. Log into Google Drive.
2. In Google Drive, make a folder named **data**, with a subfolder named
   **cornell**.
3. Visit the Cornell Movie Dialogs Corpus and download the ZIP file.
4. Unzip the file on your local machine.
5. Copy the files **movie\_lines.txt** and **movie\_conversations.txt** to the **data/cornell** folder that you
   created in Google Drive.

Now we'll need to edit the file in\_ \_Colab to point to the file on
Google Drive.

In Colab, add the following to top of the code section over the line
that begins *corpus\_name*:

::

    from google.colab import drive
    drive.mount('/content/gdrive')

Change the two lines that follow:

1. Change the **corpus\_name** value to **"cornell"**.
2. Change the line that begins with **corpus** to this:

::

    corpus = os.path.join("/content/gdrive/My Drive/data", corpus_name)

We're now pointing to the file we uploaded to Drive.

Now when you click the **Run cell** button for the code section,
you'll be prompted to authorize Google Drive and you'll get an
authorization code. Paste the code into the prompt in Colab and you
should be set.

Rerun the notebook from the **Runtime** / **Run All** menu command and
you'll see it process. (Note that this tutorial takes a long time to
run.)

Hopefully this example will give you a good starting point for running
some of the more complex tutorials in Colab. As we evolve our use of
Colab on the PyTorch tutorials site, we'll look at ways to make this
easier for users.
