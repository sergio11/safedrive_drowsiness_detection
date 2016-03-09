var express = require('express');
var quoter = require('../quoter');

var router = express.Router();

router.get('/api/random-quote', function(req, res){
	res.status(200).send(quoter.getRandomOne());
});

exports.default = router;