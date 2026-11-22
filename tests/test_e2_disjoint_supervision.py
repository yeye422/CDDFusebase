import unittest

import numpy as np

from diagnose_e2_disjoint_supervision import (
    bilinear_footprint, cross_edit_counts, overlap, region, separated,
)
from diagnose_e2_primitive_render import pixel_grid


class DisjointSupervisionTests(unittest.TestCase):
    def test_bilinear_reads_fractional_neighbors_and_exact_integer_pixel(self):
        coords = np.array([[[3.5, 4.25], [7, 2]]], dtype=np.float32)
        actual = set(map(tuple, np.argwhere(bilinear_footprint(coords, (9, 9)))))
        self.assertEqual(actual, {(4, 3), (4, 4), (5, 3), (5, 4), (2, 7)})

    def test_bilinear_probe_rejects_out_of_bounds(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            bilinear_footprint(np.array([[9.01, 7]]), (9, 10))

    def test_parallel_lines_can_have_read_overlap_even_with_distinct_centers(self):
        grid = pixel_grid((64, 64))
        target = region(np.array([[8, 24], [53, 24]]), grid)
        near = region(np.array([[8, 26], [53, 26]]), grid)
        medium = region(np.array([[8, 27], [53, 27]]), grid)
        far = region(np.array([[8, 43], [53, 43]]), grid)
        self.assertGreater(overlap(target, near)["read_read"], 0)
        self.assertFalse(separated(target, near))
        self.assertEqual(overlap(target, medium)["read_read"], 0)
        self.assertFalse(separated(target, medium))
        self.assertTrue(separated(target, far))

    def test_actual_edits_are_checked_against_other_readouts(self):
        grid = pixel_grid((64, 64))
        target = region(np.array([[8, 20], [53, 20]]), grid)
        guard = region(np.array([[8, 42], [53, 42]]), grid)
        background = np.full((64, 64), 128, np.uint8)
        clean = background.copy()
        clean[target.write] += 3
        contaminated = clean.copy()
        contaminated[guard.read] += 2
        edits = {"e2": background, "restored": clean, "smoothed": background,
                 "shifted": background.copy()}
        self.assertEqual(cross_edit_counts({"edits": edits}, target, guard, None)["restored"]["guard"], 0)
        edits["restored"] = contaminated
        self.assertEqual(cross_edit_counts({"edits": edits}, target, guard, None)["restored"]["guard"],
                         np.count_nonzero(guard.read))


if __name__ == "__main__":
    unittest.main()
