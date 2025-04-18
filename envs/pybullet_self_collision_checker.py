import numpy as np


class SelfCollisionChecker:
    def __init__(self, link_lengths):
        self.link_lengths = link_lengths
        self.num_links = len(link_lengths)

    def check_collision(self, joint_angles):
        # Get positions of each joint (including base at origin)
        points = self._compute_link_positions(joint_angles)

        # Check for intersection between all non-adjacent link pairs
        for i in range(self.num_links - 2):
            for j in range(i + 2, self.num_links):
                if self._segments_intersect(points[i], points[i+1], points[j], points[j+1]):
                    return True
        return False

    def _compute_link_positions(self, joint_angles):
        positions = [np.array([0.0, 0.0])]
        pos = np.array([0.0, 0.0])
        angle = 0.0

        for i in range(self.num_links):
            angle += joint_angles[i]
            dx = self.link_lengths[i] * np.cos(angle)
            dy = self.link_lengths[i] * np.sin(angle)
            pos = pos + np.array([dx, dy])
            positions.append(pos.copy())

        return positions

    def _segments_intersect(self, p1, p2, q1, q2):
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))
