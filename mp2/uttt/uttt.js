var pieceValues = [4, 9, 2, 3, 5, 7, 8, 1, 6];
var board = new Board();
var computerPlayer = new ComputerPlayer();
var opponent = "person";
var device = "computer";
function render() {
    $("#board").css("height", "100%").css("width", "100%");
    var c = $(window).height();
    var e = $("#boardwrapper").width();
    var b = c - 110 - 160;
    if (b > e) {
        b = e
    }
    $("#board").css("height", b + "px").css("width", b + "px");
    var d = $("#tile1").offset();
    var f = $("#tile1").width();
    var a = d.left + (f / 2) - 100;
    $("#tellResult").css("left", a + "px")
}
function startGame() {
    opponent = $("#opponent").val();
    $("#tellResult").css("display", "none").addClass("startingPosition");
    setTimeout(reshowResult, 2000);
    if (opponent != "person") {
        computerPlayer.moves = new BoardState()
    }
    board.gameType = $("#gametype").val();
    board.initialiseBoard();
    board.showNextPlayer();
    board.render();
    $("#board").off("click");
    $("#board").on("click", ".activeSquareCross", recordMove).on("click", ".activeSquareNaught", recordMove);
    $("#turnmarker").removeClass("right left").addClass("left")
}
function reshowResult() {
    $("#tellResult").css("display", "block")
}
function endGame() {
    $("#board").off("click")
}
function recordMove(c) {
    var b = c.currentTarget.dataset.tile;
    var d = c.currentTarget.dataset.square;
    var a = board.currentPlayer;
    board.placePiece(b, d, a);
    board.checkWinner(b);
    board.render();
    board.showNextPlayer();
    if (board.gameWon != "") {
        setTimeout(alertWon, 100)
    } else {
        if (opponent != "person" && board.currentPlayer == "o") {
            setTimeout(computerMakeMove, 300)
        }
    }
}
function computerMakeMove() {
    computerPlayer.makeMove();
    board.render();
    if (board.gameWon != "") {
        setTimeout(alertWon, 100)
    }
}
function alertWon() {
    board.setGameOver();
    if (board.gameWon == "x") {
        $("#resultText").text(" wins!");
        $("#resultPiece").removeClass().addClass("cross")
    } else {
        if (board.gameWon == "o") {
            $("#resultText").text(" wins!");
            $("#resultPiece").removeClass().addClass("naught")
        } else {
            $("#resultText").text("Draw");
            $("#resultPiece").removeClass().addClass("nothing")
        }
    }
    $("#tellResult").removeClass().addClass("slidein");
    setTimeout(shiftResultUp, 2000)
}
function shiftResultUp() {
    $("#tellResult").removeClass("slidein");
    setTimeout(resultsInPlace, 2000)
}
function resultsInPlace() {
    $("#tellResult").addClass("inPlace")
}
function begin() {
    $("#rulesbutton, #closerules").on("click", showHideRules);
    board.initialiseBoard();
    render();
    $(window).resize(render);
    $("#newgame").on("click", startGame)
}
$(document).ready(begin);
function showHideRules() {
    $("#ruleswrapper").toggleClass("show hide")
}
function Board() {
    this.tiles = Array();
    this.gameType = 1;
    this.lastMove = Array();
    this.currentPlayer = "x";
    this.gameWon = "";
    this.initialiseBoard = function() {
        for (var a = 0; a < 9; a++) {
            this.tiles[a] = new Tile();
            this.tiles[a].setup(a)
        }
        this.gameWon = "";
        this.lastMove[0] = -1;
        this.currentPlayer = "x"
    }
    ;
    this.checkWinner = function(d) {
        if (this.gameType == 1) {
            if (this.tiles[d].winner == this.lastMove[2]) {
                this.gameOver();
                return true
            }
            return false
        } else {
            var e = this.getScores(this.lastMove[2]);
            if (e.length < 3) {
                return false
            } else {
                var c = 0;
                var b = 1;
                var a = 2;
                while (c <= e.length - 2) {
                    b = c + 1;
                    while (b <= e.length - 1) {
                        a = b + 1;
                        while (a <= e.length) {
                            if (e[c] + e[b] + e[a] == 15) {
                                this.gameOver();
                                return true
                            }
                            a++
                        }
                        b++
                    }
                    c++
                }
            }
            return false
        }
    }
    ;
    this.getScores = function(b) {
        var c = Array();
        for (var a = 0; a < this.tiles.length; a++) {
            if (this.tiles[a].winner == b) {
                c.push(this.tiles[a].score)
            }
        }
        return c
    }
    ;
    this.placePiece = function(b, c, a) {
        this.lastMove[0] = parseInt(b);
        this.lastMove[1] = parseInt(c);
        this.lastMove[2] = a;
        this.nextPlayer();
        return this.tiles[b].placePiece(c, a)
    }
    ;
    this.render = function() {
        var c = this.whichTilesActive();
        $(".tile").removeClass("activeTile");
        for (var a = 0; a < 9; a++) {
            var b = false;
            if ($.inArray(a, c) >= 0) {
                b = true
            }
            this.tiles[a].render(b);
            if (this.tiles[a].winner == "x") {
                $("#tile" + a).addClass("winnerX")
            } else {
                if (this.tiles[a].winner == "o") {
                    $("#tile" + a).addClass("winnerO")
                } else {
                    if (b) {
                        $("#tile" + a).addClass("activeTile")
                    }
                }
            }
        }
        if (this.gameWon != "") {
            $(".tile").removeClass("activeTile")
        } else {
            if (this.lastMove[0] == -1) {
                $(".tile").removeClass("winnerX winnerO").addClass("activeTile")
            }
        }
        $("#tile" + this.lastMove[0] + "square" + this.lastMove[1]).addClass("lastmove");
        setTimeout(function() {
            $(".square").removeClass("lastmove")
        }, 100)
    }
    ;
    this.whichTilesActive = function() {
        var b = Array();
        if (this.lastMove[0] == -1) {
            b = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        } else {
            if (this.tiles[this.lastMove[1]].winner == "" && !this.tiles[this.lastMove[1]].full) {
                b.push(this.lastMove[1])
            } else {
                for (var a = 0; a < 9; a++) {
                    if (this.tiles[a].winner == "" && !this.tiles[a].full) {
                        b.push(a)
                    }
                }
            }
        }
        return b
    }
    ;
    this.nextPlayer = function() {
        if (this.currentPlayer == "x") {
            this.currentPlayer = "o"
        } else {
            this.currentPlayer = "x"
        }
    }
    ;
    this.showNextPlayer = function() {
        if (this.currentPlayer == "x") {
            $("#board").removeClass("oturn").addClass("xturn")
        } else {
            $("#board").removeClass("xturn").addClass("oturn")
        }
    }
    ;
    this.gameOver = function() {
        this.gameWon = this.lastMove[2]
    }
    ;
    this.setGameOver = function() {
        $("#board").off("click")
    }
    ;
    this.clone = function() {
        var b = new Board();
        b.gameType = this.gameType;
        b.currentPlayer = this.currentPlayer;
        b.gameWon = this.gameWon;
        b.lastMove[0] = this.lastMove[0];
        b.lastMove[1] = this.lastMove[1];
        b.lastMove[2] = this.lastMove[2];
        for (var a = 0; a < 9; a++) {
            b.tiles[a] = this.tiles[a].clone()
        }
        return b
    }
}
function Tile() {
    this.id = "";
    this.item = 0;
    this.place = 0;
    this.squares = Array();
    this.score = 0;
    this.winner = "";
    this.full = false;
    this.setup = function(b) {
        this.score = pieceValues[b];
        this.id = "tile" + b;
        this.item = $("#" + this.id);
        this.place = b;
        for (var a = 0; a < 9; a++) {
            this.squares[a] = new Square();
            this.squares[a].setup(a, b)
        }
    }
    ;
    this.placePiece = function(b, a) {
        this.squares[b].set_state(a);
        this.checkWinner(a);
        this.checkFull();
        return true
    }
    ;
    this.render = function(b) {
        for (var a = 0; a < 9; a++) {
            this.squares[a].render(b)
        }
    }
    ;
    this.checkWinner = function(d) {
        var e = this.getScores(d);
        if (e.length < 3) {
            return false
        } else {
            var c = 0;
            var b = 1;
            var a = 2;
            while (c <= e.length - 2) {
                b = c + 1;
                while (b <= e.length - 1) {
                    a = b + 1;
                    while (a <= e.length) {
                        if (e[c] + e[b] + e[a] == 15) {
                            this.winner = d;
                            return d
                        }
                        a++
                    }
                    b++
                }
                c++
            }
        }
        return false
    }
    ;
    this.checkFull = function() {
        var b = true;
        for (var a = 0; a < 9; a++) {
            if (this.squares[a].state == "") {
                b = false
            }
        }
        this.full = b
    }
    ;
    this.checkTwoInARow = function(f) {
        var d = this.getScores("");
        var g = this.getScores(f);
        var e = 0;
        if (g.length > 1) {
            var c = 0;
            var b = 1;
            var a = 0;
            while (c <= g.length - 1) {
                b = c + 1;
                while (b <= g.length) {
                    a = 0;
                    while (a <= d.length) {
                        if (g[c] + g[b] + d[a] == 15) {
                            e++;
                            d.splice(a, 1);
                            break
                        }
                        a++
                    }
                    b++
                }
                c++
            }
        }
        return e
    }
    ;
    this.emptySquares = function() {
        var b = Array();
        for (var a = 0; a < 9; a++) {
            if (this.squares[a].state == "") {
                b.push(a)
            }
        }
        return b
    }
    ;
    this.getScores = function(b) {
        var c = Array();
        for (var a = 0; a < this.squares.length; a++) {
            if (this.squares[a].state == b) {
                c.push(this.squares[a].score)
            }
        }
        return c
    }
    ;
    this.clone = function() {
        var b = new Tile();
        b.id = this.id;
        b.item = this.item;
        b.place = this.place;
        b.score = this.score;
        b.winner = this.winner;
        b.full = this.full;
        for (var a = 0; a < 9; a++) {
            b.squares[a] = this.squares[a].clone()
        }
        return b
    }
}
function Square() {
    this.state = "";
    this.id = "";
    this.item = 0;
    this.score = 0;
    this.tile = 0;
    this.setup = function(b, a) {
        this.id = "tile" + a + "square" + b;
        this.item = $("#" + this.id);
        this.score = pieceValues[b];
        this.tile = a
    }
    ;
    this.set_state = function(a) {
        if (this.state != "") {
            return false
        }
        this.state = a;
        return true
    }
    ;
    this.clear = function() {
        this.state = 0;
        return true
    }
    ;
    this.render = function(b) {
        if (this.state == "x" && !this.item.hasClass("cross")) {
            this.item.removeClass("naught cross-faded naught-faded activeSquareCross activeSquareNaught").addClass("cross")
        }
        if (this.state == "o" && !this.item.hasClass("naught")) {
            this.item.removeClass("cross cross-faded naught-faded activeSquareCross activeSquareNaught").addClass("naught")
        }
        if (this.state == "") {
            this.item.removeClass("cross cross-faded naught naught-faded activeSquareCross activeSquareNaught");
            if (b) {
                var a = board.currentPlayer;
                if (a == "x") {
                    this.item.addClass("activeSquareCross")
                } else {
                    this.item.addClass("activeSquareNaught")
                }
            }
        }
        if (board.gameWon != "") {
            this.item.removeClass("activeSquareCross activeSquareNaught")
        }
        return true
    }
    ;
    this.clone = function() {
        var a = new Square();
        a.state = this.state;
        a.id = this.id;
        a.item = this.item;
        a.score = this.score;
        a.tile = this.tile;
        return a
    }
}
function ComputerPlayer() {
    this.moves = new BoardState();
    this.analyseBoard = function() {
        var g = $.now();
        this.moves.currentState = board.clone();
        this.moves.player = "x";
        this.moves.levels = 3;
        this.moves.level = 4;
        this.moves.generateFutureStates(4);
        this.moves.analyseCurrentState();
        var a = -99999999999999;
        for (var b = 0; b < this.moves.futureStates.length; b++) {
            if (a < this.moves.futureStates[b].strength) {
                a = this.moves.futureStates[b].strength
            }
        }
        var c = $.now();
        console.log("time to think: " + (c - g));
        var e = Array();
        var f = 0;
        for (b = 0; b < this.moves.futureStates.length; b++) {
            if (this.moves.futureStates[b].strength == a) {
                f = new LegalMove();
                f.tile = this.moves.futureStates[b].currentState.lastMove[0];
                f.square = this.moves.futureStates[b].currentState.lastMove[1];
                f.piece = "o";
                e.push(f)
            }
        }
        var d = 0;
        if (e.length > 1) {
            d = Math.floor(Math.random() * (e.length - 1))
        }
        return e[d]
    }
    ;
    this.makeMove = function() {
        var a = this.analyseBoard();
        board.placePiece(a.tile, a.square, "o");
        board.checkWinner(a.tile);
        board.showNextPlayer()
    }
}
function BoardState() {
    this.currentState = 0;
    this.futureStates = Array();
    this.strength = 0;
    this.strengthWin = 0;
    this.strengthWinTile = 0;
    this.strengthff = 0;
    this.strength2iar = 0;
    this.strength2iarTile = 0;
    this.strengthb = 0;
    this.strengthInherit = 0;
    this.level = -1;
    this.levels = -1;
    this.multiplier = -1;
    this.player = "";
    this.moveMade = Array();
    this.analyseCurrentState = function() {
        var a = 0;
        if (this.currentState.gameType == 1) {
            a = this.analyseFirstTile()
        } else {
            a = this.analyse3Tiles()
        }
        return a
    }
    ;
    this.analyseFirstTile = function() {
        var c = 0;
        if (this.futureStates.length > 0) {
            var a = 0;
            if (this.player == "o") {
                c = 100000000;
                for (var b = 0; b < this.futureStates.length; b++) {
                    a = this.futureStates[b].analyseCurrentState();
                    if (a < c) {
                        c = a
                    }
                }
            } else {
                c = -1000000000;
                for (var b = 0; b < this.futureStates.length; b++) {
                    a = this.futureStates[b].analyseCurrentState();
                    if (a > c) {
                        c = a
                    }
                }
            }
        }
        this.strengthInherit = c;
        c += this.calculateWin();
        c += this.calculate2InARowFactor();
        c += this.calculateOpenFactor();
        this.strength = c;
        if (this.level == this.levels && this.strengthWin == 1000) {
            this.strength = 2000
        }
        if (this.level == 3) {
            console.log("strength: " + this.strength + "  2iar: " + this.strength2iar + "  inherit: " + this.strengthInherit + "  win: " + this.strengthWin)
        }
        return this.strength
    }
    ;
    this.analyse3Tiles = function() {
        var c = 0;
        if (this.futureStates.length > 0) {
            var a = 0;
            if (this.player == "o") {
                c = 100000000;
                for (var b = 0; b < this.futureStates.length; b++) {
                    a = this.futureStates[b].analyseCurrentState();
                    if (a < c) {
                        c = a
                    }
                }
            } else {
                c = -1000000000;
                for (var b = 0; b < this.futureStates.length; b++) {
                    a = this.futureStates[b].analyseCurrentState();
                    if (a > c) {
                        c = a
                    }
                }
            }
        }
        this.strengthInherit = c;
        c += this.calculateWin();
        c += this.calculate2InARowFactor();
        c += this.calculate2InARowTileFactor();
        c += this.calculateWinTile();
        this.strength = c;
        if (this.level == this.levels && this.strengthWin == 1000) {
            this.strength = 2000
        }
        if (this.level == 3) {
            console.log("strength: " + this.strength + "  2iar: " + this.strength2iar + "  inherit: " + this.strengthInherit + "  win: " + this.strengthWin)
        }
        return this.strength
    }
    ;
    this.generateFutureStates = function(e) {
        var c = this.getLegalMoves();
        var d = 0;
        var a = e - 1;
        this.futureStates = new Array();
        for (var b = 0; b < c.length; b++) {
            d = new BoardState();
            d.currentState = this.currentState.clone();
            d.level = a;
            d.levels = this.levels;
            d.player = this.currentState.currentPlayer;
            d.moveMade = c[b];
            d.currentState.placePiece(c[b].tile, c[b].square, c[b].piece);
            d.currentState.checkWinner(c[b].tile);
            if ((a > 0 && d.currentState.gameWon == "") && c.length < 20) {
                d.generateFutureStates(a)
            }
            this.futureStates.push(d)
        }
    }
    ;
    this.getLegalMoves = function() {
        var b = Array();
        var c = 0;
        var e = 0;
        if (this.currentState.tiles[this.currentState.lastMove[1]].winner == "" && !this.currentState.tiles[this.currentState.lastMove[1]].full) {
            e = this.currentState.tiles[this.currentState.lastMove[1]].emptySquares();
            for (var a = 0; a < e.length; a++) {
                var c = new LegalMove();
                c.tile = this.currentState.lastMove[1];
                c.square = e[a];
                c.piece = this.currentState.currentPlayer;
                b.push(c)
            }
        } else {
            for (var a = 0; a < 9; a++) {
                if (!this.currentState.tiles[a].full && this.currentState.tiles[a].winner == "") {
                    e = this.currentState.tiles[a].emptySquares();
                    for (var d = 0; d < e.length; d++) {
                        var c = new LegalMove();
                        c.tile = a;
                        c.square = e[d];
                        c.piece = this.currentState.currentPlayer;
                        b.push(c)
                    }
                }
            }
        }
        return b
    }
    ;
    this.calculateWin = function() {
        var a = 0;
        if (this.currentState.gameWon == "o") {
            a = 1000
        } else {
            if (this.currentState.gameWon == "x") {
                a = -1000
            }
        }
        this.strengthWin = a;
        return a
    }
    ;
    this.calculateWinTile = function() {
        var a = 0;
        if (this.currentState.tiles[this.currentState.lastMove[0]].winner == "x") {
            a = -500
        } else {
            if (this.currentState.tiles[this.currentState.lastMove[0]].winner == "o") {
                a = 500
            }
        }
        this.strengthWinTile = a;
        return a
    }
    ;
    this.calculate2InARowFactor = function() {
        var c = 0;
        var b = 0;
        var d = 0;
        for (var a = 0; a < 9; a++) {
            b = this.currentState.tiles[a].checkTwoInARow("x");
            d = this.currentState.tiles[a].checkTwoInARow("o");
            if (d == 1) {
                c += 10
            } else {
                if (d > 1) {
                    c += 30
                }
            }
            if (b == 1) {
                c += -10
            } else {
                if (b > 1) {
                    c += -30
                }
            }
        }
        this.strength2iar = c;
        return c
    }
    ;
    this.calculate2InARowTileFactor = function() {
        var a = 0;
        var h = 0;
        var e = 0;
        var g = Array();
        var l = Array();
        var m = Array();
        var b = Array();
        for (var f = 0; f < 9; f++) {
            g[f] = this.currentState.tiles[f].winner;
            if (this.currentState.tiles[f].winner == "") {
                b.push(f)
            } else {
                if (this.currentState.tiles[f].winner == "x") {
                    l.push(f)
                } else {
                    if (this.currentState.tiles[f].winner == "o") {
                        m.push(f)
                    }
                }
            }
        }
        var f = 0;
        var d = 0;
        var c = 0;
        if (l.length > 1) {
            f = 0;
            d = 1;
            c = 0;
            while (f <= l.length - 1) {
                d = f + 1;
                while (d <= l.length) {
                    c = 0;
                    while (c <= b.length) {
                        if (l[f] + l[d] + b[c] == 15) {
                            h += 1;
                            break
                        }
                        c++
                    }
                    d++
                }
                f++
            }
        }
        if (m.length > 1) {
            f = 0;
            d = 1;
            c = 0;
            while (f <= m.length - 1) {
                d = f + 1;
                while (d <= m.length) {
                    c = 0;
                    while (c <= b.length) {
                        if (m[f] + m[d] + b[c] == 15) {
                            e += 1;
                            break
                        }
                        c++
                    }
                    d++
                }
                f++
            }
        }
        if (e == 1) {
            a += 100
        } else {
            if (e > 1) {
                a += 300
            }
        }
        if (h == 1) {
            a += -100
        } else {
            if (h > 1) {
                a += -300
            }
        }
        this.strength2iarTile = a;
        return a
    }
    ;
    this.calculateOpenFactor = function() {
        var a = this.currentState.tiles[this.currentState.lastMove[1]].getScores("");
        var c = pieceValues[this.currentState.lastMove[1]];
        var e = 0;
        var d = 0;
        var b = 1;
        while (d <= a.length - 1) {
            b = d + 1;
            while (b <= a.length) {
                if (c + a[d] + a[b] == 15) {
                    e++
                }
                b++
            }
            d++
        }
        if (this.currentState.lastMove[2] == "x") {
            e = e * -1
        }
        return e
    }
    ;
    this.checkFutureWin = function() {
        for (var a = 0; a < this.futureStates.length; a++) {
            if (this.futureStates[a].currentState.gameWon == "x") {
                return "x"
            }
            if (this.futureStates[a].currentState.gameWon == "o") {
                return "o"
            }
        }
        return ""
    }
}
function LegalMove() {
    this.tile = -1;
    this.square = -1;
    this.piece = ""
}
var sprite = new Image();
sprite.src = "../images/sprite.png";
