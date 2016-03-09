var express = require('express');
var jwt = require('express-jwt');
var config = require('../config');
var quoter = require('../quoter');

var router = express.Router();
var jwtCheck = jwt({
	secret: config.secret
});

router.use('/api/protected', jwtCheck);

router.get('/api/protected/random-quote', function(req, res){
	res.status(200).send(quoter.getRandomOne());
});

exports.default = router;