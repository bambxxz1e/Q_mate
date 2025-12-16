// src/router/index.js
import { createRouter, createWebHistory } from "vue-router";

// 페이지 컴포넌트
import ShaderDemo from "../components/ShaderDemo.vue";
import QuestionReady from "../Question_ready.vue";
import Login from "../Login.vue";
import Interview from "../Interview.vue";
import Result from "../Result.vue";
import Join from "../Join.vue";
import MyPage from "../MyPage.vue";

const routes = [
  { path: "/", name: "home", component: ShaderDemo }, // 기본 진입점
  { path: "/question-ready", name: "question-ready", component: QuestionReady },
  { path: "/login", name: "login", component: Login },
  { path: "/join", name: "join", component: Join }, // '/Join' → '/join' 권장
  { path: "/interview", name: "interview", component: Interview },
  { path: "/result", name: "result", component: Result },
  { path: "/myPage", name: "mypage", component: MyPage },

  // 404
  { path: "/:pathMatch(.*)*", name: "not-found", component: ShaderDemo },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
  // ✅ 이 위치가 맞습니다
  scrollBehavior() {
    // 라우트 전환 시 항상 맨 위로
    return { top: 0 };
  },
});

export default router;
