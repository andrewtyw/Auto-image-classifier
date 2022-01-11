import { ElNotification } from 'element-plus';

const notice_success = function (msg) {
    ElNotification.success({
        position:"top-left",
        title: '成功',
        message: msg,
        offset:200,
        duration: 5000,
    });
};

const notice_fail = function (msg) {
    ElNotification.error({
        position:"top-left",
        title: '失败',
        message: msg,
        offset:200,
        duration: 6000,
    });
};

const getDate = function (date){
    return date.substring(0,10)+" "+date.substring(11,19)
}
// const resetForm = function (formName) {
//     this.$refs[formName].resetFields()
// };

export {notice_fail, notice_success,getDate}