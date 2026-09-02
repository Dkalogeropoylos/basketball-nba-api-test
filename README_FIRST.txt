NBA SAFE API TEST
=================

IMPORTANT:
- This is a completely separate test app.
- Do NOT replace your WNBA streamlit_app.py.
- Do NOT replace your WNBA requirements.txt.
- The WNBA model stays frozen.

EASIEST SAFE ROUTE (Streamlit Cloud):
1. Create a NEW GitHub repo, e.g. basketball-nba-api-test.
2. Upload ONLY:
   - nba_test_app.py
   - requirements.txt
3. In Streamlit Community Cloud, create a NEW app from that repo.
4. Main file path: nba_test_app.py
5. Open the app and click "Load NBA API data".
6. If it works, download the two CSV files and send them to ChatGPT.

If stats.nba.com blocks Streamlit Cloud's IP:
- We will switch the provider; your WNBA app is still untouched.

LOCAL ROUTE (optional):
1. Open PowerShell in this folder.
2. py -m pip install -r requirements.txt
3. py -m streamlit run nba_test_app.py
