<template>
  <div id="detector-container">
    <div id="title">
      <h1 style="margin-bottom: 0px">Welcome to Wall-E Depression Detector 🤖</h1>
      <div class="des">
        <p>Wall-E Depression Detector is an AI that can detect depression emotion.</p>
        <p style="top: -2px">Try to publish a post here and let Wall-E detects your emotion!👇</p>
      </div>
    </div>
    <div id="image" class="item">
      <a-space direction="vertical" :style="{ width: '100%' }">
        <a-upload draggable action="http://127.0.0.1:5000/api/upload" :fileList="file ? [file] : []"
          :show-file-list="true" @change="onChange" @progress="onProgress" class="item" @success="onUploadSuccess">
          <template #upload-button>
            <div :class="`arco-upload-list-item${file && file.status === 'error' ? ' arco-upload-list-item-error' : ''
              }`" style="width: 100%;height: 100%;display: flex;align-items: center;margin:0px">
              <div style="width: auto;height: 40vh;margin: 1px;" class="arco-upload-list-picture custom-upload-avatar"
                v-if="file && file.url">
                <img :src="file.url" style="width:100%" />
                <a-progress v-if="file.status === 'uploading' && file.percent < 100" :percent="file.percent" type="circle"
                  size="mini" :style="{
                    position: 'absolute',
                    left: '50%',
                    top: '50%',
                    transform: 'translateX(-50%) translateY(-50%)',
                  }" />
              </div>
              <div class="arco-upload-picture-card" v-else style="width: 40vh;height: 40vh;margin: 1px;">
                <div class="arco-upload-picture-card-text">
                  <IconPlus />
                  <div style="margin-top: 10px; font-weight: 600">Upload</div>
                </div>
              </div>
            </div>
          </template>
        </a-upload>
      </a-space>
    </div>
    <div id="text" class="item">
      <a-textarea placeholder="input some text..." v-model="text" id="textarea" />
    </div>
    <div id="detect" class="item">
      <a-button id='btn' type="primary" @click="onDetect">Detect</a-button>
    </div>
  </div>
</template>
<script setup lang="ts">
import { reactive, onMounted, ref } from 'vue'
import { IconEdit, IconPlus } from '@arco-design/web-vue/es/icon';
import axios from 'axios'
import request from "../utils/request";
let text = ref('')
let file_name = ''
const emit = defineEmits(['alert', 'outcome']);
const file = ref();
// 上传文件前校验
const onChange = (_: any, currentFile: any) => {
  let name = ['txt', 'csv', 'xlsx']
  let fileName = currentFile.name.split('.')
  let fileExt = fileName[fileName.length - 1]
  let isTypeOk = name.indexOf(fileExt) >= 0
  if (!isTypeOk) {
    emit('alert', 'warning', 'txt csv xlsx Supported')
  } else {
    file.value = {
      ...currentFile,
      // url: URL.createObjectURL(currentFile.file),
    };
  }

};
const onProgress = (currentFile: any) => {
  file.value = currentFile;
};
let uploaded = false
// 上传文件
const onUploadSuccess = (res: any) => {
  file_name = res.response.message
  emit('alert', 'success', 'Upload successfully!')
  uploaded = true
};

const onDetect = () => {
  if (!uploaded && text.value === '') {
    emit('alert', 'error', 'Please upload an file or input some text text')
    return
  }
  let param = new FormData(); //创建form对象
  param.append('file_name', file_name);//通过append向form对象添加数据
  param.append('text', text.value);//通过append向form对象添加数据
  request
    .post('/api/detect', param, { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } })
    .then((res) => {
      console.log(res.data)
      emit('alert', 'success', 'Detect successfully!')
      emit('outcome', res.data.message)
    })
}
</script>

<style scoped>
.des {
  margin-top: -4px;
  font-size: 1.2em;
  font-weight: 500;
  color: #999;
}

#textarea {
  width: 100%;
  height: 100%;
  border-radius: 10px;
  border: 1px dashed #d9d9d9;
  background: rgb(240, 241, 243);
  transition: border-color 0.3s;
}

#upload-btn {
  width: 100%;
  height: 100%;
  border-radius: 10px;
  border: 1px dashed #d9d9d9;
  background: rgb(240, 241, 243);
  cursor: pointer;
  transition: border-color 0.3s;
}

#btn {
  font-family: 'Roboto', sans-serif;
  background: linear-gradient(white, #e8e9ed);
  border: 1px solid #e8e9ed;
  color: black;
  font-weight: bold;
  font-size: 1.2em;
  border-radius: 10px;
  height: 2.2rem;
}

#btn:hover {
  background: linear-gradient(white, #e8eeed);
  border: 1px solid #e8e9ed;
  color: black;
  font-weight: bold;
  font-size: 1.2em;
  border-radius: 10px;
  height: 2.2rem;
}

#btn:active {
  background: radial-gradient(white, #e8eeed);
}

#detector-container {
  display: flex;
  align-items: center;
  flex-direction: column;
}

#text {
  width: 30vw;
  height: 13vh;
}

#text :first-child {
  width: 100%;
  height: 100%;
}

#detect {
  width: 20vw;
  height: 9vh;
}

#detect :first-child {
  width: 100%;
  height: 2.2rem;
}

.item {
  margin: 10px;
  width: 20vw;
  display: flex;
  justify-content: center;
}
</style>
