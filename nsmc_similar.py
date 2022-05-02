import re
import sys


def cosine_similarity(dic1, dic2):
   totalKeys = []
   for x in dic1.keys():
       if x not in totalKeys:
           totalKeys.append(x)
   for x in dic2.keys():
       if x not in totalKeys:
           totalKeys.append(x)

   denominator1 = 0
   denominator2 = 0
   numerator = 0
   for key in totalKeys:
       if key in dic1:
           denominator1 += dic1[key]
       if key in dic2:
           denominator2 += dic2[key]

       if dic1.get(key) and dic2.get(key):
           numerator += dic1[key]*dic2[key]
   denominator1 = denominator1**(1/2)
   denominator2 = denominator2**(1/2)
   return numerator/(denominator1+denominator2)


if __name__ == '__main__':
   dfIndex = 1
   koreanTFs = []
   asciiTFs = []
   otherTFs = []
   koreanDFs = {}
   asciiDFs = {}
   otherDFs = {}
   tokenIndex = {}
   docVectors = []

   f = open("ratings_train.txt", "r", encoding="UTF8")
   lines = f.readlines()
   f.close()

   for line in lines:
       tmpDic = {}
       tmpDic2 = {}
       text = "\t".join(" ".join(line.split(" ")[1:]).split("\t")[:1])
       asciiText = re.findall("\w+", text, re.ASCII)
       otherText = re.findall("[^ \uac00-\ud7a3\u0000-\u007f]+", text)
       for x in otherText:
           if x != ' ':
               if x in tmpDic:
                   tmpDic[x] += 1
               else:
                   tmpDic[x] = 1
               if x in otherDFs:
                   otherDFs[x] += 1
               else:
                   otherDFs[x] = 1
       otherTFs.append(tmpDic)
       for x in asciiText:
           if x in tmpDic2:
               tmpDic2[x] += 1
           else:
               tmpDic2[x] = 1
           if x in asciiDFs:
               asciiDFs[x] += 1
           else:
               asciiDFs[x] = 1
       asciiTFs.append(tmpDic2)
   for line in lines:
       tmpDic = {}
       line = line.replace(" ", "_")
       line = re.sub("[^가-힣_]","", line)
       line = "_"+line+"_"
       for i in range(len(line)-1):
           if line[i:i+2] in tmpDic:
               tmpDic[line[i:i+2]] += 1
           else:
               tmpDic[line[i:i+2]] = 1
           if line[i:i+2] in koreanDFs:
               koreanDFs[line[i:i+2]] += 1
           else:
               koreanDFs[line[i:i+2]] = 1
       koreanTFs.append(tmpDic)
   f = open("asciiToken.txt", "w")
   for asciiText in asciiDFs:
       f.write(asciiText + "\n")
       if asciiText not in tokenIndex:
           tokenIndex[asciiText] = dfIndex
           dfIndex += 1
   f.close()
   f = open("koreanToken.txt", "w")
   for koreanText in koreanDFs:
       f.write(koreanText + "\n")
       if koreanText not in tokenIndex:
           tokenIndex[koreanText] = dfIndex
           dfIndex += 1
   f.close()
   f = open("otherToken.txt", "w", encoding="UTF8")
   for otherText in otherDFs:
       f.write(otherText + "\n")
       if otherText not in tokenIndex:
           tokenIndex[otherText] = dfIndex
           dfIndex += 1
   f.close()

   for i in range(len(koreanTFs)):
       lineList = []
       docvect = {}
       for key in koreanTFs[i].keys():
           docvect[tokenIndex[key]] = koreanTFs[i][key]/koreanDFs[key]
       for key in asciiTFs[i].keys():
           docvect[tokenIndex[key]] = asciiTFs[i][key]/asciiDFs[key]
       for key in otherTFs[i]:
           docvect[tokenIndex[key]] = otherTFs[i][key]/otherDFs[key]
       lineList.append(lines[i][-2])
       lineList.append(docvect)
       docVectors.append(lineList)

   f = open("DocumentVector.txt", "w")
   for docVec in docVectors:
       printLine = docVec[0] + " "
       for key in docVec[1]:
           printLine += str(key) + ":" + str(docVec[1][key]) + " "
       f.write(printLine + "\n")
   f.close()

   if len(sys.argv) != 2:
       print("Input Number of TextData(1~50000)")
       sys.exit()

   testIndex = int(sys.argv[1])

   f = open("ratings_test.txt", "r", encoding="UTF8")
   testLines = f.readlines()
   testLine = testLines[testIndex]
   testPosNeg = testLine[-2]
   f.close()
   testTF = {}

   testTextReFormed = "\t".join(" ".join(testLine.split(" ")[1:]).split("\t")[:1])
   testAsciiText = re.findall("\w+", testTextReFormed, re.ASCII)
   testOtherText = re.findall("[^ \uac00-\ud7a3\u0000-\u007f]+", testTextReFormed)
   for x in testAsciiText:
       if x in asciiDFs:
           if x in testTF:
               testTF[x] += 1
           else:
               testTF[x] = 1

   for x in testOtherText:
       if x in otherDFs:
           if x in testTF:
               testTF[x] += 1
           else:
               testTF[x] = 1

   testLineKorean = testLine.replace(" ", "_")
   testLineKorean = re.sub("[^가-힣_]", "", testLineKorean)
   testLineKorean = "_" + testLineKorean + "_"

   for i in range(len(testLineKorean)-1):
       if testLineKorean[i:i+2] in koreanDFs:
           if testLineKorean[i:i+2] in testTF:
               testTF[testLineKorean[i:i+2]] += 1
           else:
               testTF[testLineKorean[i:i+2]] = 1

   testTFIDF = {}

   for key in testTF:
       if key in koreanDFs:
           testTFIDF[tokenIndex[key]] = testTF[key]/koreanDFs[key]
       if key in asciiDFs:
           testTFIDF[tokenIndex[key]] = testTF[key]/asciiDFs[key]
       if key in otherDFs:
           testTFIDF[tokenIndex[key]] = testTF[key]/otherDFs[key]

   maxSimilaritys = [0, 0, 0, 0, 0]
   maxSimIndex = [-1, -1, -1, -1, -1]
   for i in range(len(docVectors)):
       sim = cosine_similarity(testTFIDF, docVectors[i][1])
       if sim > min(maxSimilaritys[0], maxSimilaritys[1], maxSimilaritys[2], maxSimilaritys[3], maxSimilaritys[4]):
           maxSimilaritys[maxSimilaritys.index(min(maxSimilaritys[0], maxSimilaritys[1], maxSimilaritys[2], maxSimilaritys[3], maxSimilaritys[4]))] = sim
           maxSimIndex[maxSimilaritys.index(min(maxSimilaritys[0], maxSimilaritys[1], maxSimilaritys[2], maxSimilaritys[3], maxSimilaritys[4]))] = i

   print("테스트 문장 : " + testLine)
   for i in range(len(maxSimIndex)):
       print(lines[maxSimIndex[i]] + " : " + str(maxSimilaritys[i]))
