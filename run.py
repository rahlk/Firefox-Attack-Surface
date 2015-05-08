from __future__ import print_function
from __future__ import division
from os import walk
import pandas as pd
from random import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from axe.table import clone
from axe.sk import rdivDemo
from smote import SMOTE
from methods1 import createTbl


def formatData(tbl):
  Rows = [i.cells for i in tbl._rows]
  headers = [i.name for i in tbl.headers]
  return pd.DataFrame(Rows, columns=headers)


class ABCD():

  "Statistics Stuff, confusion matrix, all that jazz..."

  def __init__(self, before, after):
    self.actual = before
    self.predicted = after
    self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0
    self.abcd()

  def abcd(self):
    for a, b in zip(self.actual, self.predicted):
      if a == b:
        if b == 1:
          self.TP += 1
        else:
          self.TN += 1
      else:
        if b == 1:
          self.FP += 1
        else:
          self.FN += 1

  def all(self):
    Sen = self.TP / (self.TP + self.FN)
    Spec = self.TN / (self.TN + self.FP)
    Prec = self.TP / (self.TP + self.FP)
    Acc = (self.TP + self.TN) / (self.TP + self.FN + self.TN + self.FP)
    F1 = 2 * self.TP / (2 * self.TP + self.FP + self.FN)
    g = 2 * Sen * Spec / (Sen + Spec)
    return Sen, Spec, Prec, Acc, F1, g


class predictor():

  "CART and Random Forest of trees"

  def __init__(
          self,
          train=None,
          test=None,
          tuning=None,
          smoteit=True,
          duplicate=False):
    self.train = train
    self.test = test
    self.tuning = tuning
    self.smoteit = smoteit
    self.duplicate = duplicate

  def CART(self):
    "  CART"
    # Apply random forest Classifier to predict the number of bugs.
    if self.smoteit:
      self.train = SMOTE(
          self.train,
          atleast=1,
          atmost=300,
          resample=self.duplicate)

    clf = DecisionTreeClassifier(random_state=1)
    train_df = formatData(self.train)
    test_df = formatData(self.test)
    features = train_df.columns[:-2]
    klass = train_df[train_df.columns[-2]]
    # set_trace()
    clf.fit(train_df[features].astype('float32'), klass.astype('float32'))
    preds = clf.predict(
        test_df[test_df.columns[:-2]].astype('float32')).tolist()
    return preds

  def rforest(self):
    "  RF"
    # Apply random forest Classifier to predict the number of bugs.
    if self.smoteit:
      self.train = SMOTE(
          self.train,
          atleast=500,
          atmost=500,
          resample=self.duplicate)

    clf = RandomForestClassifier(random_state=1)
    train_df = formatData(self.train)
    test_df = formatData(self.test)
    features = train_df.columns[:-2]
    klass = train_df[train_df.columns[-2]]
    # set_trace()
    clf.fit(train_df[features].astype('float32'), klass.astype('float32'))
    preds = clf.predict(
        test_df[test_df.columns[:-2]].astype('float32')).tolist()
    return preds


class main():

  " Main Class"

  def __init__(self, directory='./Data/'):
    self.dir = directory

  def file2pandas(self, File):
    fread = open(File, 'r')
    rows = [line for line in fread]
    head = rows[0].strip().split(',')  # Get the headers
    body = [[1 if r == 'Y' else 0 if r == 'N' else r
             for r in row.strip().split(',')] for row in rows[1:]]
    return pd.DataFrame(body, columns=head)

  def formatData(self, tbl):
    Rows = [i.cells for i in tbl._rows]
    headers = [i.name for i in tbl.headers]
    return pd.DataFrame(Rows, columns=headers)

  def explorer2(self):
    files = [filenames for (_, __, filenames) in walk(self.dir)][0]
    for f in files:
      return [self.dir + f]

  def flatten(self, x):
    """
    Takes an N times nested list of list like [[a,b],[c, [d, e]],[f]]
    and returns a single list [a,b,c,d,e,f]
    """
    result = []
    for el in x:
      if hasattr(el, "__iter__") and not isinstance(el, basestring):
        result.extend(self.flatten(el))
      else:
        result.append(el)
    return result

  def kFoldCrossVal(self, data, k=5, smote = False):
    acc = []
    sen = []
    spec = []
    prec = []
    f = []
    g = []
    chunks = lambda l, n: [l[i:i + n] for i in range(0, len(l), int(n))]
    rows = data._rows
    shuffle(rows)
    sqe = chunks(rows, int(len(rows) / k))
    if len(sqe) > k:
      sqe = sqe[:-2] + [sqe[-2] + sqe[-1]]
    for indx in xrange(k):
      testRows = sqe.pop(indx)
      trainRows = self.flatten([s for s in sqe if not s in testRows])
      train, test = clone(data, rows=[
          i.cells for i in trainRows]), clone(data, rows=[
              i.cells for i in testRows])
      test_df = formatData(test)
      actual = test_df[
          test_df.columns[-2]].astype('float32').tolist()
      before = predictor(train=train, test=test, smoteit=smote).rforest()
      acc.append(ABCD(before=actual, after=before).all()[3])
      sen.append(ABCD(before=actual, after=before).all()[0])
      spec.append(ABCD(before=actual, after=before).all()[1])
      prec.append(ABCD(before=actual, after=before).all()[2])
      f.append(ABCD(before=actual, after=before).all()[-2])
      g.append(ABCD(before=actual, after=before).all()[-1])
      sqe.insert(k, testRows)
    return acc, sen, spec, prec, f, g

  def crossval(self, _s=True, k=2):
    cv_acc = ['            Accuracy']
    cv_prec = ['           Precision']
    cv_sen = ['Sensitivity (Recall)']
    cv_spec = ['         Specificity']
    cv_f = ['                   f']
    cv_g = ['                   g']
    for _ in xrange(k):
      proj = self.explorer2()
      data = createTbl(proj, isBin=False, _smote=False)
      a, b, c, d, e, f = self.kFoldCrossVal(data, k=k, smote=_s)
      cv_acc.extend(a)
      cv_sen.extend(b)
      cv_spec.extend(c)
      cv_prec.extend(d)
      cv_f.extend(e)
      cv_g.extend(f)
    return cv_acc, cv_sen, cv_spec, cv_prec, cv_f, cv_g


def _doCrossVal():

  for smote in [True, False]:
    cv_acc = []
    cv_sen = []
    cv_spec = []
    cv_prec = []
    cv_f = []
    cv_g = []

    acc, sen, spec, prec, f, g = main().crossval(_s=smote, k=10)

    cv_acc += acc
    cv_sen += sen
    cv_spec += spec
    cv_prec += prec
    cv_f += f
    cv_g += g

    print("## After Sampling") if smote else print("## Before Sampling")
    rdivDemo([cv_prec,
              cv_sen,
              cv_spec,
              cv_acc,
              cv_f,
              cv_g],
             isLatex=False)
  rdivDemo(cv_spec, isLatex=False)

if __name__ == '__main__':
  _doCrossVal()
