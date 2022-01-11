<template >
  <div style="background-color: #2b88ec;height: 40px;margin-left: -8px;margin-right: -8px;margin-top: -8px;">
    <div style="float: left;font-weight: bold;margin-left: 20px;font-size: 25px;margin-top: 2px">

      <div style="color:#ffffff">
        深潜鱼类识别
        <span style="color:#ffffff;font-size: 9px;float: right">
          19计科1 王天寅 python项目
        </span>
      </div >

    </div>
    <div style="float: left;margin-left: 10px;margin-top: 4px">
<!--      <el-button type="success" circle size="mini" style="height: 20px" @click="randomChange"><i class="el-icon-refresh-right"></i></el-button>-->
      <el-tooltip
          class="item"
          effect="dark"
          placement="bottom"
      >
        <template #content> 随机图片(用户上传的)
          <br />
          <br />只支持以下鱼的分类:
          <br />Angelfish
          <br />Aulostomidae
          <br />Batfish
          <br />Butterflyfish
          <br />Centriscidae
          <br />Coralfish
          <br />Fistularidae
          <br />Garden_Eel
          <br />lizardfishes
          <br />Moray_Eel
          <br />Parrotfish
          <br />Pineapplefish
          <br />Scorpaenidae
          <br />Snake_Eel
          <br />Soldierfish
          <br />Splendid_Toadfish
          <br />Squirrelfish
          <br />Striped_Eel_Catfish
          <br />Synanceiidae
          <br />Tetrarogidae

        </template>
        <el-button type="success" round size="small" style="height: 20px" @click="randomChange"><i class="el-icon-refresh-right">随机</i></el-button>
      </el-tooltip>
    </div>
  </div>
  <div class="content-box">
    <el-image :src="item" :fit="fit" style="width: 100%;height: 100%; filter: blur(20px);overflow: hidden "></el-image>
  </div>
  <div >

    <div class="banner">
      <div style="margin-top: 20px">

        <el-card>

          <img class="carouselImg" ref="imgH" :src='item' style="width:100%;height:auto" alt="" @load="imgLoad">
          <el-table v-if="tableData.length!==0" :data="tableData" size="mini" v-loading="loading">
            <el-table-column prop="cata" label="种类" />
            <el-table-column prop="p" label="概率" />
            <el-table-column label="bing图片" width="100" >
              <template #default="scope">
                <el-button size="mini" type="primary" @click="go(scope.row)" >bing图片</el-button>
              </template>
            </el-table-column>
          </el-table>
          <div style="margin-top: 20px">
            <el-upload
                ref="uploadImg"
                class="upload-demo"
                action="http://120.24.230.237:5000/up_photo"
                :auto-upload="true"
                :before-upload="beforeUpload"
                :on-success="fileUploadSuccess"
                :limit='1'
            >
              <template #trigger>
                <el-button size="small" type="primary" @click="clean">
                  <span style="font-size: 15px;margin: 20px">上传图片</span>
                </el-button>
              </template>
              <el-button
                  style="margin-left: 10px"
                  size="small"
                  type="success"
                  @click="getRes"
              >
                <span style="font-size: 15px;margin: 20px">点击分类</span>

              </el-button>
              <template #tip>
                <div class="el-upload__tip">
                  <span style="font-size: 10px;color: #101110">图片大小请不要超过1MB</span>
                </div>
              </template>
            </el-upload>


          </div>
        </el-card>
      </div>

    </div>
  </div>


</template>

<script>
import {notice_fail, notice_success} from "@/api/elementApi";
import request from "@/utils/request";

export default {
  created() {

    // document.getElementById("banner_bg").style.paddingTop = '10.734%'
  },
  data() {
    return {
      item: 'http://120.24.230.237:5000/show/2021121221215051.jpg',
      ifUploaded:false,
      tableData: [],
      loading:false,
      fit: "fill"
    }
  },
  methods: {
    randomChange(){
      this.loading=true
      this.tableData = []
      request.get("/random").then(res=>{
        this.item = res.data.path_show
        this.loading=false
      })
    },
    beforeUpload(file) {
      console.log(file.type)
      const isJPG = (file.type === 'image/png') || (file.type === 'image/jpeg')
      const isLt1M = file.size / 1024 / 1024 < 1
      if (!isJPG) {
        notice_fail('上传头像图片只能是 jpg/png 格式!')
      }
      if (!isLt1M) {
        notice_fail("上传图片大小不能超过 1MB!")
      }
      return isJPG && isLt1M
    },
    fileUploadSuccess(res) {
      this.ifUploaded = true   // 标记上传了图片
      console.log("res:", res)
      // this.tempCover = res.data
      this.item = res.path_dl
      console.log("res.data", res.data)
    },
    getRes() {
      this.loading=true
      var form = {"url": this.item}
      request.post("/getres", form).then(res => {
        console.log("res", res)
        this.tableData = res.data.res
        this.loading = false
      })
    },
    clean(){
      if(this.ifUploaded){
        this.$refs['uploadImg'].clearFiles();
      }

    },
    go(row){
      var url = "https://www.bing.com/images/search?q="+row.cata
      window.open(url)
    }
  }
}
</script>

<style>
.main {
  width: 100%;
  height: 100%;
}

.banner {
  width: 40%;

  position: absolute;
  top: 50%;
  left: 50%;
  -webkit-transform: translate(-50%, -50%);
  -moz-transform: translate(-50%, -50%);
  -ms-transform: translate(-50%, -50%);
  -o-transform: translate(-50%, -50%);
  transform: translate(-50%, -50%);
}

#banner_bg {
  width: 100%; /*宽度铺满屏幕*/
  padding-top: 52.734%; /*图片高度除以宽度，得到此值*/
  background: url('http://localhost:5000/show/2021121216213615.jpg'); /*两个center分别为水平和垂直方向的对齐方式*/
  background-repeat: no-repeat;
  background-size: 100% 100%; /*背景水平铺满*/

  filter: blur(20px); /*虚化值，越大越模糊*/
}

.banner-contain {
  position: absolute; /*设置内容层绝对定位*/
  width: 100%;
  text-align: center;
  z-index: 6; /*将内容至于上层*/
  margin-top: 0%;
}

.bg_bur {
  background: url('./assets/Coralfish.61.jpg');
  background-size: 100% 100%;
  /*float: left;*/
  position: absolute;
  background-attachment: fixed;
  background-repeat: no-repeat;
  z-index: -1;
  filter: blur(0px);
}

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}

#nav {

}

#nav a {
  font-weight: bold;
  color: #2c3e50;
}

.content-box {
  position: absolute;
  left: 0px;
  right: 0;
  top: 50px;
  bottom: 0;
  background: #ffffff;
}

.center {
  position: absolute;
  top: 50%;
  left: 50%;
  -webkit-transform: translate(-50%, -50%);
  -moz-transform: translate(-50%, -50%);
  -ms-transform: translate(-50%, -50%);
  -o-transform: translate(-50%, -50%);
  transform: translate(-50%, -50%);
}

#nav a.router-link-exact-active {
  color: #42b983;
}
</style>
