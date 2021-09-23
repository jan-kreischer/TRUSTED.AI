=== HEROKU ===
1. Create new heroku project
heroku create <app-name>

2. Open heroku app
heroku open

3. Check logs
heroku logs --tail

=== ERRORS & FIXES ===
“Your account has reached its concurrent builds limit”

$ heroku plugins:install heroku-builds
$ heroku builds:cancel
$ heroku restart
