from math import e
from ForestConverter import TreeConverter
import numpy as np
import heapq

import re


class HybridTreeConverter(TreeConverter):
    def __init__(self, dim, namespace, featureType):
        super().__init__(dim, namespace, featureType)

    def getArrayLenType(self, arrLen):
        arrayLenBit = int(np.log2(arrLen)) + 1
        if arrayLenBit <= 8:
            arrayLenDataType = "unsigned char"
        elif arrayLenBit <= 16:
            arrayLenDataType = "unsigned short"
        else:
            arrayLenDataType = "unsigned int"
        return arrayLenDataType

    def getImplementation(self, head, treeID):
        raise NotImplementedError(
            "This function should not be called directly, but only by a sub-class")

    def getHeader(self, splitType, treeID, arrLen, numClasses):
        dimBit = int(np.log2(self.dim)) + 1 if self.dim != 0 else 1

        if dimBit <= 8:
            dimDataType = "unsigned char"
        elif dimBit <= 16:
            dimDataType = "unsigned short"
        else:
            dimDataType = "unsigned int"

        featureType = self.getFeatureType()
        if (numClasses == 2):
            headerCode = """struct {namespace}_Node{treeID} {
                        //bool isLeaf;
                        //unsigned int prediction;
                        {dimDataType} feature;
                        {splitType} split;
                        {arrayLenDataType} leftChild;
                        {arrayLenDataType} rightChild;
                        unsigned char indicator;

                };\n""".replace("{namespace}", self.namespace) \
                .replace("{treeID}", str(treeID)) \
                .replace("{splitType}", splitType) \
                .replace("{dimDataType}", dimDataType) \
                .replace("{arrayLenDataType}", self.getArrayLenType(arrLen))
        else:
            headerCode = """struct {namespace}_Node{treeID} {
                           //bool isLeaf;
                            {dimDataType} feature;
                            {splitType} split;
                            {arrayLenDataType} leftChild;
                            {arrayLenDataType} rightChild;
                            unsigned char indicator;
                };\n""".replace("{namespace}", self.namespace) \
                .replace("{treeID}", str(treeID)) \
                .replace("{splitType}", splitType) \
                .replace("{dimDataType}", dimDataType) \
                .replace("{arrayLenDataType}", self.getArrayLenType(arrLen))

        headerCode += "inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n" \
            .replace("{treeID}", str(treeID)) \
            .replace("{dim}", str(self.dim)) \
            .replace("{namespace}", self.namespace) \
            .replace("{feature_t}", featureType)
        return headerCode

    def getCode(self, tree, treeID, numClasses):
        # kh.chen
        # Note: this function has to be called once to traverse the tree to calculate the probabilities.
        tree.getProbAllPaths()
        cppCode, arrLen = self.getImplementation(tree.head, treeID)

        if self.containsFloat(tree):
            splitDataType = "float"
        else:
            lower, upper = self.getSplitRange(tree)

            bitUsed = 0
            if lower > 0:
                prefix = "unsigned"
                maxVal = upper
            else:
                prefix = ""
                bitUsed = 1
                maxVal = max(-lower, upper)

            splitBit = int(np.log2(maxVal) + 1 if maxVal != 0 else 1)

            if splitBit <= (8-bitUsed):
                splitDataType = prefix + " char"
            elif splitBit <= (16-bitUsed):
                splitDataType = prefix + " short"
            else:
                splitDataType = prefix + " int"
        headerCode = self.getHeader(splitDataType, treeID, arrLen, numClasses)

        return headerCode, cppCode


import ctypes
def float_to_bin(number):
    return "{0:#0{1}x}".format( (ctypes.c_uint.from_buffer(ctypes.c_float(number)).value) ,10)
def double_to_bin(number):
    return "{0:#0{1}x}".format( (ctypes.c_ulong.from_buffer(ctypes.c_double(number)).value) , 18)

def byte_width(dtype):
    if dtype==" char" or dtype=="unsigned char":
        return 1
    if dtype==" short" or dtype=="unsigned short":
        return 2
    if dtype==" int" or dtype=="unsigned int":
        return 4
    if dtype==" long" or dtype=="unsigned long":
        return 8
    if dtype=="float":
        return 4
    if dtype=="double":
        return 8

class RCITHybridTreeConverter(HybridTreeConverter):
    def __init__(self, dim, namespace, featureType, architecture, segment_depth, start_native):
        super().__init__(dim, namespace, featureType)
        self.segment_depth = segment_depth
        self.start_native = start_native
        self.architecture = architecture

    def getHeader(self, splitType, treeID, arrLen, numClasses):
        dimBit = int(np.log2(self.dim)) + 1 if self.dim != 0 else 1

        if dimBit <= 8:
            dimDataType = "unsigned char"
        elif dimBit <= 16:
            dimDataType = "unsigned short"
        else:
            dimDataType = "unsigned int"

        featureType = self.getFeatureType()
        headerCode = "#define SPLIT_DATATYPE_SIZE "+str(byte_width(splitType))+"\n"
        if splitType == "float" or splitType == "double":
            headerCode += "#define SPLIT_DATATYPE_FLOATING \n"
        if splitType == "unsigned char" or splitType == "unsigned short" or splitType== "unsigned int" or splitType == "unsigned long" or splitType == "float" or splitType == "double":
            headerCode += "#define SPLIT_DATATYPE_UNSIGNED \n"

        headerCode += "inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n" \
            .replace("{treeID}", str(treeID)) \
            .replace("{dim}", str(self.dim)) \
            .replace("{namespace}", self.namespace) \
            .replace("{feature_t}", featureType)

        return headerCode

    def getImplementation(self, head, treeID):

        label_idx=[0]

        #Generate pure if else tree
        def write_tree(head, indent="", label_idx=label_idx):
            if head.prediction is not None:
                #Leaf
                #Move return value to return register and jump to end
                code=""
                if self.architecture == "intel":
                    code= indent+"\"mov $"+str(int(np.argmax(head.prediction)))+", %0;\"\n"
                    code+= indent+"\"jmp __rtitt_end_"+str(treeID)+";\"\n"
                else:
                    code= indent+"\"mov %0, #"+str(int(np.argmax(head.prediction)))+";\"\n"
                    code+= indent+"\"b __rtitt_end_"+str(treeID)+";\"\n"
                return code

            scalarext=False
            code=""

            #Normal node
            #Load feature value
            if self.getFeatureType() == "float":
                code = indent+"#ifdef SPLIT_DATATYPE_FLOATING\n"
                code += indent+"#if SPLIT_DATATYPE_SIZE == 4\n"
                if self.architecture == "intel":
                    code += indent+"\"movss "+str(head.feature*4)+"(%1), %%xmm1;\"\n"
                else:
                    code += indent+"\"ldr s1, [%1, "+str(head.feature*4)+"];\"\n"
                code += indent+"#elif SPLIT_DATATYPE_SIZE == 8\n"
                if self.architecture == "intel":
                    code += indent+"\"movss "+str(head.feature*4)+"(%1), %%xmm3;\"\n"
                    code += indent+"\"cvtss2sd %%xmm3, %%xmm1;\"\n"
                else:
                    code += indent+"\"ldr s3, [%1, "+str(head.feature*4)+"];\"\n"
                    code += indent+"\"fcvt d1, s3;\"\n"
                code += indent+"#endif\n"
                code += indent+"#endif\n"
            elif self.getFeatureType() == "double":
                code = indent+"#ifdef SPLIT_DATATYPE_FLOATING\n"
                code += indent+"#if SPLIT_DATATYPE_SIZE == 8\n"
                if self.architecture == "intel":
                    code += indent+"\"movsd "+str(head.feature*8)+"(%1), %%xmm1;\"\n"
                else:
                    code += indent+"\"ldr d1, [%1, "+str(head.feature*8)+"];\"\n"
                code += indent+"#elif SPLIT_DATATYPE_SIZE == 4\n"
                if self.architecture == "intel":
                    code += indent+"\"movsd "+str(head.feature*8)+"(%1), %%xmm3;\"\n"
                    code += indent+"\"cvtsd2ss %%xmm3, %%xmm1;\"\n"
                else:
                    code += indent+"\"ldr d3, [%1, "+str(head.feature*8)+"];\"\n"
                    code += indent+"\"fcvt s1, d3;\"\n"
                code += indent+"#endif\n"
                code += indent+"#endif\n"
            elif self.getFeatureType() == " char":
                if self.architecture == "intel":
                    code = indent+"\"mov "+str(head.feature*1)+"(%1), %%cl;\"\n"
                    code += indent+"\"movsx %%cl, %%rbx;\"\n"
                else:
                    code = indent+"\"ldrsb x1, [%1, "+str(head.feature*1)+"];\"\n"
                scalarext=True
            elif self.getFeatureType() == "unsigned char":
                if self.architecture == "intel":
                    code = indent+"\"mov "+str(head.feature*1)+"(%1), %%bl;\"\n"
                else:
                    code = indent+"\"ldrb x1, [%1, "+str(head.feature*1)+"];\"\n"
                scalarext=True
            elif self.getFeatureType() == " short":
                if self.architecture == "intel":
                    code = indent+"\"mov "+str(head.feature*2)+"(%1), %%cx;\"\n"
                    code += indent+"\"movsx %%cx, %%rbx;\"\n"
                else:
                    code = indent+"\"ldrsh x1, [%1, "+str(head.feature*2)+"];\"\n"
                scalarext=True
            elif self.getFeatureType() == "unsigned short":
                if self.architecture == "intel":
                    code = indent+"\"mov "+str(head.feature*2)+"(%1), %%bx;\"\n"
                else:
                    code = indent+"\"ldrh x1, [%1, "+str(head.feature*2)+"];\"\n"
                scalarext=True
            elif self.getFeatureType() == " int":
                if self.architecture == "intel":
                    code = indent+"\"mov "+str(head.feature*4)+"(%1), %%ecx;\"\n"
                    code += indent+"\"movsx %%ecx, %%rbx;\"\n"
                else:
                    code = indent+"\"ldrsw x1, [%1, "+str(head.feature*4)+"];\"\n"
                scalarext=True
            elif self.getFeatureType() == "unsigned int":
                if self.architecture == "intel":
                    code = indent+"\"mov "+str(head.feature*4)+"(%1), %%ebx;\"\n"
                else:
                    code = indent+"\"ldrw x1, [%1, "+str(head.feature*4)+"];\"\n"
                scalarext=True
            elif self.getFeatureType() == " long":
                if self.architecture == "intel":
                    code = indent+"\"mov "+str(head.feature*8)+"(%1), %%rbx;\"\n"
                else:
                    code = indent+"\"ldrs x1, [%1, "+str(head.feature*8)+"];\"\n"
                scalarext=True
            elif self.getFeatureType() == "unsigned long":
                if self.architecture == "intel":
                    code = indent+"\"mov "+str(head.feature*8)+"(%1), %%rbx;\"\n"
                else:
                    code = indent+"\"ldr x1, [%1, "+str(head.feature*8)+"];\"\n"
                scalarext=True

            if scalarext==True:
                code += indent+"#ifdef SPLIT_DATATYPE_FLOATING\n"
                code += indent+"#if SPLIT_DATATYPE_SIZE == 8\n"
                if self.architecture == "intel":
                    code += indent+"\"cvtsi2sd %%rbx, %%xmm1;\"\n"
                else:
                    code += indent+"\"scvtf d1, x1;\"\n"
                code += indent+"#elif SPLIT_DATATYPE_SIZE == 4\n"
                if self.architecture == "intel":
                    code += indent+"\"cvtsi2sd %%rbx, %%xmm3;\"\n"
                    code += indent+"\"cvtsd2ss %%xmm3, %%xmm1;\"\n"
                else:
                    code += indent+"\"scvtf s1, x1;\"\n"
                code += indent+"#endif\n"
                code += indent+"#endif\n"

            code += indent+"//="+str(head.split)+"\n"
            code += indent+"#ifdef SPLIT_DATATYPE_FLOATING\n"
            code += indent+"#if SPLIT_DATATYPE_SIZE == 4\n"
            if self.architecture == "intel":
                code += indent+"\"mov $"+float_to_bin(head.split)+", %%eax;\"\n"
                #code += indent+"\"mov %%eax, (%2);\"\n"
                code += indent+"\"movd %%eax, %%xmm2;\"\n"
                #Do actual comparison
                code += indent+"\"comiss %%xmm1, %%xmm2;\"\n"
            else:
                bin_string=float_to_bin(head.split)
                code += indent+"//\"mov w2, #"+float_to_bin(head.split)+";\"\n"
                code += indent+"\"movz w2, #0x"+bin_string[-4:]+";\"\n"
                code += indent+"\"movk w2, #0x"+bin_string[-8:-4]+", lsl 16;\"\n"
                #code += indent+"\"mov %%eax, (%2);\"\n"
                code += indent+"\"fmov s2, w2;\"\n"
                #Do actual comparison
                code += indent+"\"fcmp s1, s2;\"\n"
            code += indent+"#elif SPLIT_DATATYPE_SIZE == 8\n"
            if self.architecture == "intel":
                code += indent+"\"mov $"+double_to_bin(head.split)+", %%rax;\"\n"
                #code += indent+"\"mov %%eax, (%2);\"\n"
                code += indent+"\"movq %%rax, %%xmm2;\"\n"
                #Do actual comparison
                code += indent+"\"comisd %%xmm1, %%xmm2;\"\n"
            else:
                bin_string=double_to_bin(head.split)
                code += indent+"//\"mov x2, #"+double_to_bin(head.split)+";\"\n"
                code += indent+"\"movz w2, #0x"+bin_string[-4:]+";\"\n"
                code += indent+"\"movk w2, #0x"+bin_string[-8:-4]+", lsl 16;\"\n"
                code += indent+"\"movk w2, #0x"+bin_string[-12:-8]+", lsl 32;\"\n"
                code += indent+"\"movk w2, #0x"+bin_string[-16:-12]+", lsl 48;\"\n"
                #code += indent+"\"mov %%eax, (%2);\"\n"
                code += indent+"\"fmov d2, x2;\"\n"
                #Do actual comparison
                code += indent+"\"fcmp d1, d2;\"\n"
            code += indent+"#endif\n"
            code += indent+"#else\n"
            if self.architecture == "intel":
                code += indent+"\"mov $"+str(int(head.split))+", %%rax;\"\n"
                #Do actual comparison
                code += indent+"\"cmp %%rbx, %%rax;\"\n"
            else:
                code += indent+"\"mov x2, #"+str(int(head.split))+";\"\n"
                #Do actual comparison
                code += indent+"\"cmp x1, x2;\"\n"
            code += indent+"#endif\n"

            swap=head.probRight > head.probLeft
            label=label_idx[0]
            label_idx[0]+=1
            if self.architecture == "intel":
                code += indent+"#ifdef SPLIT_DATATYPE_UNSIGNED\n"
                code += indent+"\"j"+("n" if swap else "")+"b __rtitt_lab_"+str(label)+"_"+str(treeID)+";\"\n"
                code += indent+"#else\n"
                code += indent+"\"j"+("n" if swap else "")+"l __rtitt_lab_"+str(label)+"_"+str(treeID)+";\"\n"
                code += indent+"#endif\n"
            else:
                code += indent+"\"b."+("le" if swap else "gt")+" __rtitt_lab_"+str(label)+"_"+str(treeID)+";\"\n"

            code += write_tree((head.rightChild if swap else head.leftChild), indent+"   ")

            #Define else branch
            code += indent+"\"__rtitt_lab_"+str(label)+"_"+str(treeID)+":\"\n"
            code += write_tree((head.leftChild if swap else head.rightChild), indent+"   ")
            return code

        ifelsecode=write_tree(head)

        cppCode = """
                    inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]){
                            volatile unsigned int result;
                            float splittarget=0;
                            asm volatile(
                            \n{ifelsecode}\n
                            "__rtitt_end_{treeID}:"
                            """.replace("{treeID}", str(treeID)) \
                                .replace("{dim}", str(self.dim)) \
                                .replace("{ifelsecode}", ifelsecode) \
                                .replace("{namespace}", self.namespace) \
                                .replace("{feature_t}", self.getFeatureType())

        if self.architecture == "intel":
            cppCode += """:"=r"(result) :"r"(pX),"r"(&splittarget):"xmm1", "xmm2","xmm3","rax","rbx","rcx");"""
        else:
            cppCode += """:"=r"(result) :"r"(pX),"r"(&splittarget):"d1", "d2","d3","x1","x2","x3");"""

        cppCode += """
                            return result;
                        }
                    """

        return cppCode, 0