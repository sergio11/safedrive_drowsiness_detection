var logger = require('morgan')
var cors = require('cors');
var http = require('http');
var express = require('express');
var errorhandler = require('errorhandler');
var dotenv = require('dotenv');
var bodyParser = require('body-parser');

//routes
var user_routes = require('./routes/user-routes');
var anonymous_routes = require('./routes/anonymous-routes');
var protected_routes = require('./routes/protected-routes');

var app = express();

dotenv.load();

// Parsers
// old version of line
// app.use(bodyParser.urlencoded());
// new version of line
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(cors());

app.use(function(err, req, res, next){
  if (err.name === 'StatusError') {
    res.send(err.status, err.message);
  } else {
    next(err);
  }
});

if (process.env.NODE_ENV === 'development') {
  app.use(express.logger('dev'));
  app.use(errorhandler())
}

app.use(anonymous_routes.default);
app.use(protected_routes.default);
app.use(user_routes.default);

var port = process.env.PORT || 3001;

http.createServer(app).listen(port,function(err){
  console.log('listening in http://localhost:' + port);
});