# TODO: Fix Animal Disease Chatbot to Use Dynamic Predictions

## Steps to Complete

- [x] Update chatbot/views.py:
  - [x] Import JsonResponse from django.http
  - [x] Fix predict_disease function: Add back clean_text call and compute confidence
  - [x] Add static dictionary for disease-specific recommendations
  - [x] Update chatbot_view to handle AJAX requests and return JSON response

- [x] Update chatbot/templates/chatbot.html:
  - [x] Modify JavaScript to use fetch for AJAX submission
  - [x] Dynamically display bot response based on backend JSON
  - [x] Ensure chat interface updates correctly (scroll, clear input)

- [x] Test the changes:
  - [x] Run server and verify varied predictions for different inputs
  - [x] Check for any errors in console or logs
