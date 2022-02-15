import dim
from ..operate import Operate
from ..constant import Constant

class PowOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(PowOperate,self).__init__(left,right,"pow",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    if (self.catch and self._grads.get(self.left.name,None)):
      part4 = self._grads[self.left.name]
    else:
      c = Constant(self.right.eval() - 1)
      part2 = dim.autograd.PowOperate.wrapper(self.left,c)
      part3 = dim.autograd.MulOperate.wrapper(self.right,part2)
      part4 = dim.autograd.MulOperate.wrapper(part3,prevOp)
      self._grads[self.left.name] = part4
      
    rst = self.left.partGrad(partial,part4)
    self._grads[partial.name]=rst
    return rst  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst=self.left.expression() + "^" + self.right.expression()
    self._expressionStr = rst
    return rst
  
  def eval(self,useCatch=True):
    if (useCatch and self.catch and self._data is not None): return self._data
    rst= self.left.eval(useCatch)**self.right.eval(useCatch)
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    if right.type!="Constant":raise Exception("pow right must be a number constant")
    right=right.data[0]
    if (right==0): return Constant(1)
    if (right==1): return left
    '''
    if (left.type=="Constant" and (left.data==1).all()): return Constant(1)
    '''
    return PowOperate(left,right,args,name)
