"""Tests that play the game off-screen with unicode3d's playtest. Run them with python -m unittest.

Each test plays a fresh Game: keys in, then a check of the words on screen (expect) or of the game's
state (game.current is the screen showing). Add one for each feature.
"""
import tempfile
import unittest

from unicode3d.playtest import Playtest

from game import ROUND_TIME, Game, GameOver, Play, Title


class GameTests(unittest.TestCase):
    def play(self, size=(24, 80)):
        """A fresh game (always the same round: seed 1) and a playtest of it, size as (rows, cols)."""
        out = tempfile.TemporaryDirectory()     # pictures from shot() go here, and are thrown away
        self.addCleanup(out.cleanup)
        self.game = Game(seed=1)
        return Playtest(self.game, size=size, out=out.name)

    def test_title_screen(self):
        test = self.play()
        test.wait(3)
        test.expect("[ Play ]")
        self.assertIsInstance(self.game.current, Title)

    def test_enter_starts_a_round(self):
        test = self.play()
        test.press("enter")
        test.wait(2)
        self.assertIsInstance(self.game.current, Play)
        test.expect("score 0")

    def test_clicking_play_starts_a_round(self):
        test = self.play()
        test.wait(1)                            # the button is where the last frame drew it
        test.click_text("Play")
        test.wait(1)
        self.assertIsInstance(self.game.current, Play)

    def test_arrows_move_the_player(self):
        test = self.play()
        test.press("enter")
        test.hold("right", frames=15)           # half a second
        x, _, z = self.game.current.player.position
        self.assertGreater(x, 1.0)
        self.assertAlmostEqual(z, 0.0)

    def test_walking_onto_a_gem_takes_it(self):
        test = self.play()
        test.press("enter")
        play = self.game.current
        gem = play.gems[0]
        gem.position = (1.0, 0.5, 0.0)          # put a gem in the player's way
        test.hold("right", frames=10)
        self.assertEqual(play.score, 1)
        self.assertNotEqual(gem.position, (1.0, 0.5, 0.0))   # it went somewhere new
        test.expect("score 1")

    def test_the_round_ends_when_time_is_up(self):
        test = self.play()
        test.press("enter")
        test.wait(round(ROUND_TIME * 30) + 2)   # 30 frames a second
        self.assertIsInstance(self.game.current, GameOver)
        test.expect("GAME OVER")
        test.press("enter")                     # Again
        self.assertIsInstance(self.game.current, Play)

    def test_esc_goes_back_to_the_title_and_q_quits(self):
        test = self.play()
        test.press("enter")
        test.press("esc")
        self.assertIsInstance(self.game.current, Title)
        test.press("q")
        self.assertTrue(test.quit)

    def test_plays_cleanly(self):
        test = self.play()
        test.wait(5)
        test.press("enter")
        test.hold("up", "left", frames=20)
        test.resize(12, 40)                     # a small terminal mustn't break anything
        test.wait(5)
        report = test.report()
        self.assertEqual(report.warnings, [])
        self.assertEqual(report.blank_frames, [])


if __name__ == "__main__":
    unittest.main()
