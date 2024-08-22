//var m =new Matrix(3,2)
class Matrix
{
  constructor(rows,cols)
  {
    this.rows=rows;
    this.cols=cols;
    this.data=[];

    for (var i = 0; i < this.rows; i++)
    {
      this.data[i]=[];
      for (var j = 0; j < this.cols; j++)
      {
        this.data[i][j]=0;
      }
    }
  }

static multiply(a,b)
{
  if (a.cols!=b.rows)
  {
    console.log("Coloumns of A must match Row of B");
    return undefined;
  }

  var result=new Matrix(a.rows,b.cols)

  for (var i = 0; i < result.rows; i++)
  {
    for (var j = 0; j < result.cols; j++)
    {
      let sum=0;
      for (var k = 0; k < a.cols; k++)
      {
        /*
          a[i][j]*b[i][j]+
          a[i][j+1]*b[i+1][j]+
          a[i][j+2]*b[i+2][j]

        */
        sum+=a.data[i][k]*b.data[k][j];
      }
      result.data[i][j]=sum;

    }
  }
  return result;
}

multiply(n)
{
  if (n instanceof Matrix)
  {
    // hadamard product
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] *= n.data[i][j];
      }
    }
  }
  else 
  {
    // Scalar product
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] *= n;
      }
    }
  }
}

add(n)
  {
    if ( n instanceof Matrix)
    {
      for (var i = 0; i < this.rows; i++)
      {
        for (var j = 0; j < this.cols; j++)
        {
          this.data[i][j]=this.data[i][j]+n.data[i][j];
        }
      }
    }
    else
    {
      for (var i = 0; i < this.rows; i++)
      {
        for (var j = 0; j < this.cols; j++)
        {
          this.data[i][j]=this.data[i][j]+n;
        }
      }
    }
  }

  randomize()
  {
    for (var i = 0; i < this.rows; i++)
    {
      for (var j = 0; j < this.cols; j++)
      {
        this.data[i][j]=Math.floor(Math.random()*2-1)
      }
    }
  }

  static transpose(matrix)
  {
    var result=new Matrix(matrix.cols,matrix.rows);
    for (let i = 0; i < matrix.rows; i++)
    {
      for (let j = 0; j < matrix.cols; j++)
      {
        result.data[j][i]=matrix.data[i][j];
      }
    }

    return result;
  }

  print(){
    console.table(this.data)
  }

  map(fn)
  {
    for (let i = 0; i < this.rows; i++)
    {
      for (let j = 0; j < this.cols; j++)
      {
        let val=this.data[i][j];
        this.data[i][j]=fn(val);
      }
    }
  }

  static map(matrix,fn)
  {
    let result=new Matrix(matrix.rows,matrix.cols);

    for (let i = 0; i < matrix.rows; i++)
    {
      for (let j = 0; j < matrix.cols; j++)
      {
        let val=matrix.data[i][j];
        result.data[i][j]=fn(val);
      }
    }
    return result;
  }

  convertMatrixToArray()
  {
    let arr=[];
    for (let i = 0; i < this.rows; i++)
    {
      for (let j = 0; j < this.cols; j++)
      {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  }

  static convertArrayToMatrix(inputArray)
  {
    var result=new Matrix(inputArray.length,1);
    for (var i = 0; i < inputArray.length; i++)
    {
      result.data[i][0]=inputArray[i];
    }

    return result;
  }

  static substract(a,b)
  {
    let results=new Matrix(a.rows,a.cols);
    for (let i = 0; i < results.rows; i++)
    {
      for (let j = 0; j < results.cols; j++)
      {
        results.data[i][j]=a.data[i][j]-b.data[i][j];
      }
    }
    return results;
  }

}
