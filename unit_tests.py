#!/usr/bin/python

'''
Martin Kersner, m.kersner@gmail.com
2015/11/30

Unit tests for eval_segm.py.
'''

import numpy as np
import eval_segm as es
import unittest

class pixel_accuracy_UnitTests(unittest.TestCase):
    '''
    Wrong inputs
    '''
    def test1dInput(self):
        mat = np.array([0])
        self.assertRaises(IndexError, es.pixel_accuracy, mat, mat)

    def testDiffDim(self):
        mat0 = np.array([[0,0], [0,0]])
        mat1 = np.array([[0,0,0], [0,0,0]])
        self.assertRaisesRegexp(es.EvalSegErr, "DiffDim", es.pixel_accuracy, mat0, mat1)

    '''
    Correct inputs
    '''
    def testOneClass(self):
        segm = np.array([[0,0], [0,0]])
        gt   = np.array([[0,0], [0,0]])

        res = es.pixel_accuracy(segm, gt)
        self.assertEqual(res, 1.0)

    def testTwoClasses0(self):
        segm = np.array([[1,1,1,1,1], [1,1,1,1,1]])
        gt   = np.array([[0,0,0,0,0], [0,0,0,0,0]])

        res = es.pixel_accuracy(segm, gt)
        self.assertEqual(res, 0)

    def testTwoClasses1(self):
        segm = np.array([[1,0,0,0,0], [0,0,0,0,0]])
        gt   = np.array([[0,0,0,0,0], [0,0,0,0,0]])

        res = es.pixel_accuracy(segm, gt)
        self.assertEqual(res, (9.0)/(10.0))

    def testTwoClasses2(self):
        segm = np.array([[0,0,0,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,0,0,0,0], [0,0,0,0,0]])

        res = es.pixel_accuracy(segm, gt)
        self.assertEqual(res, (9.0+0.0)/(9.0+1.0))

    def testThreeClasses0(self):
        segm = np.array([[0,0,0,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,2,0,0,0], [0,0,0,0,0]])

        res = es.pixel_accuracy(segm, gt)
        self.assertEqual(res, (8.0+0.0+0.0)/(8.0+1.0+1.0))

    def testThreeClasses1(self):
        segm = np.array([[0,2,0,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,0,0,0,0], [0,0,0,0,0]])

        res = es.pixel_accuracy(segm, gt)
        self.assertEqual(res, (8.0+0.0)/(9.0+1.0))

    def testFourClasses0(self):
        segm = np.array([[0,2,3,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,0,0,0,0], [0,0,0,0,0]])

        res = es.pixel_accuracy(segm, gt)
        self.assertEqual(res, (7.0+0.0)/(9.0+1.0))

    def testFourClasses1(self):
        segm = np.array([[1,2,3,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,0,0,0,0], [0,0,0,0,0]])

        res = es.pixel_accuracy(segm, gt)
        self.assertEqual(res, (7.0+1.0)/(9.0+1.0))

    def testFiveClasses0(self):
        segm = np.array([[1,2,3,4,3], [0,0,0,0,0]])
        gt   = np.array([[1,0,3,0,0], [0,0,0,0,0]])

        res = es.pixel_accuracy(segm, gt)
        self.assertEqual(res, (5.0+1.0+1.0)/(8.0+1.0+1.0))

class mean_accuracy_UnitTests(unittest.TestCase):
    '''
    Wrong inputs
    '''
    def test1dInput(self):
        mat = np.array([0])
        self.assertRaises(IndexError, es.mean_accuracy, mat, mat)

    def testDiffDim(self):
        mat0 = np.array([[0,0], [0,0]])
        mat1 = np.array([[0,0,0], [0,0,0]])
        self.assertRaisesRegexp(es.EvalSegErr, "DiffDim", es.mean_accuracy, mat0, mat1)

    '''
    Correct inputs
    '''
    def testOneClass(self):
        segm = np.array([[0,0], [0,0]])
        gt   = np.array([[0,0], [0,0]])

        res = es.mean_accuracy(segm, gt)
        self.assertEqual(res, 1.0)

    def testTwoClasses0(self):
        segm = np.array([[1,1,1,1,1], [1,1,1,1,1]])
        gt   = np.array([[0,0,0,0,0], [0,0,0,0,0]])

        res = es.mean_accuracy(segm, gt)
        self.assertEqual(res, 0)

    def testTwoClasses1(self):
        segm = np.array([[1,0,0,0,0], [0,0,0,0,0]])
        gt   = np.array([[0,0,0,0,0], [0,0,0,0,0]])

        res = es.mean_accuracy(segm, gt)
        self.assertEqual(res, 9.0/10.0)

    def testTwoClasses2(self):
        segm = np.array([[0,0,0,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,0,0,0,0], [0,0,0,0,0]])

        res = es.mean_accuracy(segm, gt)
        self.assertEqual(res, np.mean([9.0/9.0, 0.0/1.0]))

    def testThreeClasses0(self):
        segm = np.array([[0,0,0,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,2,0,0,0], [0,0,0,0,0]])

        res = es.mean_accuracy(segm, gt)
        self.assertEqual(res, np.mean([8.0/8.0, 0.0/1.0, 0.0/1.0]))

    def testThreeClasses1(self):
        segm = np.array([[0,2,0,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,0,0,0,0], [0,0,0,0,0]])

        res = es.mean_accuracy(segm, gt)
        self.assertEqual(res, np.mean([8.0/9.0, 0.0/1.0]))

    def testFourClasses0(self):
        segm = np.array([[0,2,3,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,0,0,0,0], [0,0,0,0,0]])

        res = es.mean_accuracy(segm, gt)
        self.assertEqual(res, np.mean([7.0/9.0, 0.0/1.0]))

    def testFourClasses1(self):
        segm = np.array([[1,2,3,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,0,0,0,0], [0,0,0,0,0]])

        res = es.mean_accuracy(segm, gt)
        self.assertEqual(res, np.mean([7.0/9.0, 1.0/1.0]))

    def testFiveClasses0(self):
        segm = np.array([[1,2,3,4,3], [0,0,0,0,0]])
        gt   = np.array([[1,0,3,0,0], [0,0,0,0,0]])

        res = es.mean_accuracy(segm, gt)
        self.assertEqual(res, np.mean([5.0/8.0, 1.0, 1.0]))

class mean_IU_UnitTests(unittest.TestCase):
    '''
    Wrong inputs
    '''
    def test1dInput(self):
        mat = np.array([0])
        self.assertRaises(IndexError, es.mean_IU, mat, mat)

    def testDiffDim(self):
        mat0 = np.array([[0,0], [0,0]])
        mat1 = np.array([[0,0,0], [0,0,0]])
        self.assertRaisesRegexp(es.EvalSegErr, "DiffDim", es.mean_IU, mat0, mat1)

    '''
    Correct inputs
    '''
    def testOneClass(self):
        segm = np.array([[0,0], [0,0]])
        gt   = np.array([[0,0], [0,0]])

        res = es.mean_IU(segm, gt)
        self.assertEqual(res, 1.0)

    def testTwoClasses0(self):
        segm = np.array([[1,1,1,1,1], [1,1,1,1,1]])
        gt   = np.array([[0,0,0,0,0], [0,0,0,0,0]])

        res = es.mean_IU(segm, gt)
        self.assertEqual(res, 0)

    def testTwoClasses1(self):
        segm = np.array([[1,0,0,0,0], [0,0,0,0,0]])
        gt   = np.array([[0,0,0,0,0], [0,0,0,0,0]])

        res = es.mean_IU(segm, gt)
        self.assertEqual(res, np.mean([0.9]))

    def testTwoClasses2(self):
        segm = np.array([[0,0,0,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,0,0,0,0], [0,0,0,0,0]])

        res = es.mean_IU(segm, gt)
        self.assertEqual(res, np.mean([0.9, 0]))

    def testThreeClasses0(self):
        segm = np.array([[0,0,0,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,2,0,0,0], [0,0,0,0,0]])

        res = es.mean_IU(segm, gt)
        self.assertEqual(res, np.mean([8.0/10.0, 0, 0]))

    def testThreeClasses1(self):
        segm = np.array([[0,2,0,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,0,0,0,0], [0,0,0,0,0]])

        res = es.mean_IU(segm, gt)
        self.assertEqual(res, np.mean([8.0/10.0, 0]))

    def testFourClasses0(self):
        segm = np.array([[0,2,3,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,0,0,0,0], [0,0,0,0,0]])

        res = es.mean_IU(segm, gt)
        self.assertEqual(res, np.mean([7.0/10.0, 0]))

    def testFourClasses1(self):
        segm = np.array([[1,2,3,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,0,0,0,0], [0,0,0,0,0]])

        res = es.mean_IU(segm, gt)
        self.assertEqual(res, np.mean([7.0/9.0, 1]))

    def testFiveClasses0(self):
        segm = np.array([[1,2,3,4,3], [0,0,0,0,0]])
        gt   = np.array([[1,0,3,0,0], [0,0,0,0,0]])

        res = es.mean_IU(segm, gt)
        self.assertEqual(res, np.mean([5.0/8.0, 1, 1.0/2.0]))

class frequency_weighted_IU_UnitTests(unittest.TestCase):
    '''
    Wrong inputs
    '''
    def test1dInput(self):
        mat = np.array([0])
        self.assertRaises(IndexError, es.frequency_weighted_IU, mat, mat)

    def testDiffDim(self):
        mat0 = np.array([[0,0], [0,0]])
        mat1 = np.array([[0,0,0], [0,0,0]])
        self.assertRaisesRegexp(es.EvalSegErr, "DiffDim", es.frequency_weighted_IU, mat0, mat1)

    '''
    Correct inputs
    '''
    def testOneClass(self):
        segm = np.array([[0,0], [0,0]])
        gt   = np.array([[0,0], [0,0]])

        res = es.frequency_weighted_IU(segm, gt)
        self.assertEqual(res, 1.0)

    def testTwoClasses0(self):
        segm = np.array([[1,1,1,1,1], [1,1,1,1,1]])
        gt   = np.array([[0,0,0,0,0], [0,0,0,0,0]])

        res = es.frequency_weighted_IU(segm, gt)
        self.assertEqual(res, 0)

    def testTwoClasses1(self):
        segm = np.array([[1,0,0,0,0], [0,0,0,0,0]])
        gt   = np.array([[0,0,0,0,0], [0,0,0,0,0]])

        res = es.frequency_weighted_IU(segm, gt)
        self.assertEqual(res, (1.0/10.0)*(10.0*9.0/10.0))

    def testTwoClasses2(self):
        segm = np.array([[0,0,0,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,0,0,0,0], [0,0,0,0,0]])

        res = es.frequency_weighted_IU(segm, gt)
        # Almost equal!
        self.assertAlmostEqual(res, (1.0/10.0)*((9.0*9.0/10.0)+(1.0*0.0/1.0))) 

    def testThreeClasses0(self):
        segm = np.array([[0,0,0,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,2,0,0,0], [0,0,0,0,0]])

        res = es.frequency_weighted_IU(segm, gt)
        # Almost equal!
        self.assertAlmostEqual(res, (1.0/10.0)*((8.0*8.0/10.0)+(1.0*0.0/1.0)+(1.0*0.0/1.0)))

    def testThreeClasses1(self):
        segm = np.array([[0,2,0,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,0,0,0,0], [0,0,0,0,0]])

        res = es.frequency_weighted_IU(segm, gt)
        # Almost equal!
        self.assertAlmostEqual(res, (1.0/10.0)*((9.0*8.0/10.0)+(1.0*0.0/1.0)))

    def testFourClasses0(self):
        segm = np.array([[0,2,3,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,0,0,0,0], [0,0,0,0,0]])

        res = es.frequency_weighted_IU(segm, gt)
        self.assertEqual(res, (1.0/10.0)*((9.0*7.0/10.0)+(1.0*0.0/1.0)))

    def testFourClasses1(self):
        segm = np.array([[1,2,3,0,0], [0,0,0,0,0]])
        gt   = np.array([[1,0,0,0,0], [0,0,0,0,0]])

        res = es.frequency_weighted_IU(segm, gt)
        self.assertEqual(res, (1.0/10.0)*((9.0*7.0/9.0)+(1.0*1.0/1.0)))

    def testFiveClasses0(self):
        segm = np.array([[1,2,3,4,3], [0,0,0,0,0]])
        gt   = np.array([[1,0,3,0,0], [0,0,0,0,0]])

        res = es.frequency_weighted_IU(segm, gt)
        self.assertEqual(res, (1.0/10.0)*((8.0*5.0/8.0)+(1.0*1.0/1.0)+(1.0*1.0/2.0)))


if __name__ == "__main__":
    unittest.main()
