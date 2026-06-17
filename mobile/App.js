import React, {
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  Platform,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

import AsyncStorage from "@react-native-async-storage/async-storage";
import * as FileSystem from "expo-file-system";

import * as ImagePicker from "expo-image-picker";
import * as DocumentPicker from "expo-document-picker";
import { useVideoPlayer, VideoView } from "expo-video";

const BACKEND_URL =
  "http://formcheck-ai-api-v3.eba-pvfk7qtv.us-west-2.elasticbeanstalk.com";

const API_URL = Platform.OS === "web" ? "/api" : BACKEND_URL;

const MEDIA_URL = Platform.OS === "web" ? "/api" : BACKEND_URL;

const fullUrl = (path) => {
  if (!path) return null;
  if (String(path).startsWith("http")) return path;
  return `${MEDIA_URL}${path}`;
};

const formatLabel = (v) =>
  v
    ? String(v)
        .replaceAll("_", " ")
        .replace(/\b\w/g, (c) => c.toUpperCase())
    : "N/A";

const getStatusColor = (status) => {
  if (
    [
      "poor",
      "incomplete",
      "drifting",
      "severe_flare",
      "limited_range",
    ].includes(status)
  ) {
    return "#ef4444";
  }

  if (
    [
      "borderline",
      "needs_work",
      "possible",
      "shallow",
      "fair",
      "possibly_shallow",
    ].includes(status)
  ) {
    return "#f59e0b";
  }

  return "#22c55e";
};

const getScoreColor = (score) => {
  if (score >= 9) return "#22c55e";
  if (score >= 7.5) return "#84cc16";
  if (score >= 6) return "#f59e0b";
  return "#ef4444";
};

const getBestRep = (reps) => {
  if (!reps || reps.length === 0) return null;

  return reps.reduce((best, rep) => {
    return Number(rep.score || 0) > Number(best.score || 0) ? rep : best;
  }, reps[0]);
};

const getPhaseConfig = (exerciseLabel) => {
  const label = String(exerciseLabel || "").toLowerCase();

  if (label.includes("thruster")) {
  return {
    title: "Thruster Phase Review",
    text: "Squat/Dip → Drive → Lockout",
    items: [
      ["dip", "Squat/Dip"],
      ["drive", "Drive"],
      ["lockout", "Lockout"],
    ],
  };
}
  
  if (label.includes("clean & jerk") || label.includes("clean and jerk")) {
    return {
      title: "Clean & Jerk Phase Review",
      text: "Setup → Clean Catch → Jerk Dip → Jerk Drive → Jerk Catch → Finish",
      items: [
        ["setup", "Setup"],
        ["clean_catch", "Clean Catch"],
        ["jerk_dip", "Jerk Dip"],
        ["jerk_drive", "Jerk Drive"],
        ["jerk_catch", "Jerk Catch"],
        ["finish", "Finish"],
      ],
    };
  }
  
  if (label.includes("split jerk")) {
  return {
    title: "Split Jerk Phase Review",
    text: "Setup → Dip → Drive → Catch → Finish",
    items: [
      ["setup", "Setup"],
      ["dip", "Dip"],
      ["drive", "Drive"],
      ["catch", "Catch"],
      ["finish", "Finish"],
    ],
  };
}

  if (label.includes("olympic") || label.includes("clean") || label.includes("snatch") || label.includes("jerk")) {
    return {
      title: "Olympic Lift Phase Review",
      text: "Setup → First Pull → Extension → Catch → Finish",
      items: [
        ["setup", "Setup"],
        ["first_pull", "First Pull"],
        ["extension", "Extension"],
        ["catch", "Catch"],
        ["finish", "Finish"],
      ],
    };
  }

  if (label.includes("push press")) {
    return {
      title: "Push Press Phase Review",
      text: "Setup → Dip → Drive → Catch → Lockout",
      items: [
        ["setup", "Setup"],
        ["dip", "Dip"],
        ["drive", "Drive"],
        ["catch", "Catch"],
        ["lockout", "Lockout"],
      ],
    };
  }

  if (label.includes("squat")) {
    return {
      title: "Squat Phase Review",
      text: "Setup → Descent → Bottom → Ascent → Lockout",
      items: [
        ["setup", "Setup"],
        ["descent", "Descent"],
        ["bottom", "Bottom"],
        ["ascent", "Ascent"],
        ["lockout", "Lockout"],
      ],
    };
  }

  if (label.includes("bench")) {
    return {
      title: "Bench Press Phase Review",
      text: "Setup → Descent → Bottom → Press → Lockout",
      items: [
        ["setup", "Setup"],
        ["descent", "Descent"],
        ["bottom", "Bottom"],
        ["press", "Press"],
        ["lockout", "Lockout"],
      ],
    };
  }

  if (label.includes("pull")) {
  return {
    title: "Pull-Up Phase Review",
    text: "Hang → Pull → Top → Finish",
    items: [
      ["hang", "Hang"],
      ["pull", "Pull"],
      ["top", "Top"],
      ["finish", "Finish"],
    ],
  };
}

  if (label.includes("muscle")) {
    return {
      title: label.includes("ring")
        ? "Ring Muscle-Up Phase Review"
        : "Bar Muscle-Up Phase Review",
      text: "Hang → Pull → Transition → Dip → Lockout → Finish",
      items: [
        ["hang", "Hang"],
        ["pull", "Pull"],
        ["transition", "Transition"],
        ["dip", "Dip"],
        ["lockout", "Lockout"],
        ["finish", "Finish"],
      ],
    };
  }

  if (label.includes("handstand push")) {
    return {
      title: "Handstand Push-Up Phase Review",
      text: "Setup → Descent → Bottom → Ascent → Lockout",
      items: [
        ["setup", "Setup"],
        ["descent", "Descent"],
        ["bottom", "Bottom"],
        ["ascent", "Ascent"],
        ["lockout", "Lockout"],
      ],
    };
  }

  if (label.includes("push up") || label.includes("push-up")) {
    return {
      title: "Push-Up Phase Review",
      text: "Setup → Descent → Bottom → Ascent → Lockout",
      items: [
        ["setup", "Setup"],
        ["descent", "Descent"],
        ["bottom", "Bottom"],
        ["ascent", "Ascent"],
        ["lockout", "Lockout"],
      ],
    };
  }

  if (label.includes("burpee")) {
    return {
      title: "Burpee Phase Review",
      text: "Hands Down → Plank → Jump In → Stand → Finish",
      items: [
        ["start", "Hands Down"],
        ["hands_down", "Plank"],
        ["plank", "Jump In"],
        ["stand", "Stand"],
        ["finish", "Finish"],
      ],
    };
  }

  return {
    title: "Deadlift Phase Review",
    text: "Setup → Pull → Mid → Finish → Lockout",
    items: [
      ["setup", "Setup"],
      ["pull", "Pull"],
      ["mid", "Mid"],
      ["finish", "Finish"],
      ["lockout", "Lockout"],
    ],
  };
};

const getInteractiveZones = (result) => {
  const label = String(result?.exercise_label || "").toLowerCase();
  const bestRep = getBestRep(result?.rep_feedback || []);
  const breakdown = bestRep?.breakdown || {};

    if (label.includes("thruster")) {
  return [
    {
      id: "squat_dip",
      title: "Squat/Dip",
      imageKey: "dip",
      status: breakdown.squat_depth || "good",
      note: "Use a full squat before driving the bar overhead.",
    },
    {
      id: "drive",
      title: "Drive",
      imageKey: "drive",
      status: breakdown.torso_stack || "good",
      note: "Stay tall and drive through the bar.",
    },
    {
      id: "lockout",
      title: "Lockout",
      imageKey: "lockout",
      status: breakdown.lockout || breakdown.active_finish || "good",
      note: "Finish fully locked out overhead.",
    },
  ];
}

  if (label.includes("burpee")) {
    return [
      {
        id: "hands_down",
        title: "Hands Down",
        imageKey: "start",
        status: breakdown.start || "good",
        note: "Move quickly to the floor while staying balanced.",
      },
      {
        id: "plank",
        title: "Plank",
        imageKey: "hands_down",
        status: breakdown.hands_down || "good",
        note: "Keep your body tight and avoid sagging through the core.",
      },
      {
        id: "jump_in",
        title: "Jump In",
        imageKey: "plank",
        status: breakdown.plank || "good",
        note: "Bring your feet underneath you efficiently.",
      },
      {
        id: "stand",
        title: "Stand",
        imageKey: "stand",
        status: breakdown.stand || "good",
        note: "Stand tall with control before finishing the rep.",
      },
      {
        id: "finish",
        title: "Finish",
        imageKey: "finish",
        status: breakdown.finish || "good",
        note: "Complete the rep fully before starting the next one.",
      },
    ];
  }

  if (label.includes("pull")) {
      return [
        {
          id: "hang",
          title: "Hang",
          imageKey: "hang",
          status: "good",
          note: "Start from a controlled dead hang.",
        },
        {
          id: "pull",
          title: "Pull",
          imageKey: "pull",
          status: "good",
          note: "Pull strongly with control.",
        },
        {
          id: "top",
          title: "Top",
          imageKey: "top",
          status: "good",
          note: "Finish high with chin near or above the bar.",
        },
        {
          id: "finish",
          title: "Finish",
          imageKey: "finish",
          status: "good",
          note: "Return to a controlled hang before the next rep.",
        },
      ];
    }

  if (label.includes("muscle")) {
    return [
      {
        id: "pull",
        title: "Pull",
        imageKey: "pull",
        status: breakdown.pull || "good",
        note: "Pull high before starting the transition.",
      },
      {
        id: "transition",
        title: "Transition",
        imageKey: "transition",
        status: breakdown.transition || "good",
        note: "Turn over aggressively and keep the rings or bar close.",
      },
      {
        id: "support",
        title: "Support",
        imageKey: "dip",
        status: breakdown.support || "good",
        note: "Stabilize above the bar or rings before finishing.",
      },
      {
        id: "lockout",
        title: "Lockout",
        imageKey: "lockout",
        status: breakdown.lockout || "good",
        note: "Finish tall with strong locked-out arms.",
      },
    ];
  }

  if (label.includes("handstand push")) {
    return [
      {
        id: "depth",
        title: "Depth",
        imageKey: "bottom",
        status: breakdown.depth || "good",
        note: "Lower your head toward the floor under control.",
      },
      {
        id: "body_line",
        title: "Body Line",
        imageKey: "descent",
        status: breakdown.body_line || "good",
        note: "Keep your body stacked and avoid arching or sagging.",
      },
      {
        id: "press",
        title: "Press",
        imageKey: "ascent",
        status: breakdown.control || "good",
        note: "Press smoothly away from the floor.",
      },
      {
        id: "lockout",
        title: "Lockout",
        imageKey: "lockout",
        status: breakdown.lockout || "good",
        note: "Finish with arms fully locked out overhead.",
      },
    ];
  }

  if (label.includes("push up")) {
    return [
      {
        id: "depth",
        title: "Depth",
        imageKey: "bottom",
        status: breakdown.depth || "good",
        note: "Lower your chest closer to the floor.",
      },
      {
        id: "body_line",
        title: "Body Line",
        imageKey: "descent",
        status: breakdown.body_line || "good",
        note: "Keep shoulders, hips, and ankles aligned.",
      },
      {
        id: "press",
        title: "Press",
        imageKey: "ascent",
        status: breakdown.control || "good",
        note: "Press smoothly back to lockout.",
      },
      {
        id: "lockout",
        title: "Lockout",
        imageKey: "lockout",
        status: breakdown.lockout || "good",
        note: "Finish with elbows nearly straight.",
      },
    ];
  }

  if (label.includes("push press")) {
    return [
      {
        id: "dip",
        title: "Dip",
        imageKey: "dip",
        status: breakdown.dip || "good",
        note: "Use a vertical dip and drive straight through the bar.",
      },
      {
        id: "drive",
        title: "Drive",
        imageKey: "drive",
        status: breakdown.bar_path || "good",
        note: "Keep the bar close and drive straight overhead.",
      },
      {
        id: "catch",
        title: "Catch",
        imageKey: "catch",
        status: breakdown.control || "good",
        note: "Catch the bar stacked over your shoulders and midfoot.",
      },
      {
        id: "lockout",
        title: "Lockout",
        imageKey: "lockout",
        status: breakdown.lockout || "good",
        note: "Finish with elbows fully extended overhead.",
      },
    ];
  }

  if (label.includes("squat")) {
    return [
      {
        id: "torso",
        title: "Torso",
        imageKey: "descent",
        status: breakdown.torso || "good",
        note: "Keep your chest tall and avoid folding forward.",
      },
      {
        id: "depth",
        title: "Depth",
        imageKey: "bottom",
        status: breakdown.depth || "good",
        note: "Reach clear depth while keeping control.",
      },
      {
        id: "knees",
        title: "Knees",
        imageKey: "bottom",
        status: breakdown.knees || "good",
        note: "Drive knees out and keep them tracking over toes.",
      },
      {
        id: "lockout",
        title: "Lockout",
        imageKey: "lockout",
        status: breakdown.lockout || "good",
        note: "Stand tall and finish the rep under control.",
      },
    ];
  }

  if (label.includes("bench")) {
    return [
      {
        id: "wrists",
        title: "Wrists",
        imageKey: "press",
        status: breakdown.wrists || "good",
        note: "Keep wrists stacked and avoid letting them bend back.",
      },
      {
        id: "elbows",
        title: "Elbows",
        imageKey: "bottom",
        status: breakdown.elbows || "good",
        note: "Keep elbows controlled without excessive flare.",
      },
      {
        id: "bar",
        title: "Bar Path",
        imageKey: "press",
        status: breakdown.bar_path || breakdown.depth || "good",
        note: "Move from chest to lockout with a controlled path.",
      },
      {
        id: "lockout",
        title: "Lockout",
        imageKey: "lockout",
        status: breakdown.lockout || "good",
        note: "Finish with arms fully extended.",
      },
    ];
  }

  if (label.includes("split jerk")) {
  return [
    { id: "setup", title: "Setup", imageKey: "setup", status: "good", note: "Start tall and braced with the bar in the front rack." },
    { id: "dip", title: "Dip", imageKey: "dip", status: "good", note: "Dip straight down with a vertical torso." },
    { id: "drive", title: "Drive", imageKey: "drive", status: "good", note: "Drive aggressively through the legs." },
    { id: "catch", title: "Catch", imageKey: "catch", status: "good", note: "Catch locked out overhead in a strong split." },
    { id: "finish", title: "Finish", imageKey: "finish", status: "good", note: "Recover under control and stabilize overhead." },
  ];
}
  
  if (label.includes("clean & jerk") || label.includes("clean and jerk")) {
    return [
      {
        id: "setup",
        title: "Setup",
        imageKey: "setup",
        status: "good",
        note: "Start tight with the bar close and chest up.",
      },
      {
        id: "clean_catch",
        title: "Clean Catch",
        imageKey: "clean_catch",
        status: "good",
        note: "Receive the clean in a strong front rack.",
      },
      {
        id: "jerk_dip",
        title: "Jerk Dip",
        imageKey: "jerk_dip",
        status: "good",
        note: "Dip straight down with a vertical torso.",
      },
      {
        id: "jerk_drive",
        title: "Jerk Drive",
        imageKey: "jerk_drive",
        status: "good",
        note: "Drive powerfully through the legs.",
      },
      {
        id: "jerk_catch",
        title: "Jerk Catch",
        imageKey: "jerk_catch",
        status: "good",
        note: "Catch locked out overhead with control.",
      },
      {
        id: "finish",
        title: "Finish",
        imageKey: "finish",
        status: "good",
        note: "Stand tall and stabilize the finished position.",
      },
    ];
  }

  if (
    label.includes("olympic") ||
    label.includes("clean") ||
    label.includes("snatch") ||
    label.includes("jerk")
  ) {
    return [
      {
        id: "setup",
        title: "Setup",
        imageKey: "setup",
        status: "good",
        note: "Start tight with the bar close and chest up.",
      },
      {
        id: "first_pull",
        title: "First Pull",
        imageKey: "first_pull",
        status: "good",
        note: "Keep the bar close as it passes the knees.",
      },
      {
        id: "extension",
        title: "Extension",
        imageKey: "extension",
        status: "good",
        note: "Drive tall through the legs and hips.",
      },
      {
        id: "catch",
        title: "Catch",
        imageKey: "catch",
        status: "good",
        note: "Receive the bar under control.",
      },
      {
        id: "finish",
        title: "Finish",
        imageKey: "finish",
        status: "good",
        note: "Stand tall and stabilize the finished position.",
      },
    ];
  }

  return [
    {
      id: "back",
      title: "Back Position",
      imageKey: "pull",
      status: breakdown.back || "good",
      note: "Brace hard and keep a neutral spine.",
    },
    {
      id: "hips",
      title: "Hip Hinge",
      imageKey: "mid",
      status: breakdown.hinge || "good",
      note: "Push hips back and keep tension through the pull.",
    },
    {
      id: "bar",
      title: "Bar Path",
      imageKey: "mid",
      status: breakdown.bar_path || "good",
      note: "Keep the bar close to your body.",
    },
    {
      id: "lockout",
      title: "Lockout",
      imageKey: "lockout",
      status: breakdown.lockout || "good",
      note: "Finish tall with hips and knees extended.",
    },
  ];
};

const QUEUE_KEY = "formcheck_pending_videos";

const HISTORY_KEY = "formcheck_analysis_history";

const savePendingVideo = async (video) => {
  const id = Date.now().toString();

  const ext = video.uri?.split(".").pop() || "mov";
  const localUri = `${FileSystem.documentDirectory}pending_${id}.${ext}`;

  await FileSystem.copyAsync({
    from: video.uri,
    to: localUri,
  });

  const item = {
    id,
    uri: localUri,
    name: video.name || video.fileName || `Workout ${id}`,
    createdAt: new Date().toISOString(),
    status: "pending",
  };

  const existing =
    JSON.parse(await AsyncStorage.getItem(QUEUE_KEY)) || [];

  const updated = [item, ...existing];

  await AsyncStorage.setItem(
    QUEUE_KEY,
    JSON.stringify(updated)
  );

  return item;
};

const loadPendingVideos = async (setter) => {
  const existing =
    JSON.parse(await AsyncStorage.getItem(QUEUE_KEY)) || [];

  setter(existing);
};

const saveAnalysisHistory = async (analysis) => {
  try {
    const item = {
      id: Date.now().toString(),
      createdAt: new Date().toISOString(),
      exercise_label: analysis.exercise_label,
      confidence: analysis.confidence,
      set_summary: analysis.set_summary || {},
      rep_feedback: analysis.rep_feedback || [],
      coaching_zones: analysis.coaching_zones || [],
    };

    const existing =
      JSON.parse(await AsyncStorage.getItem(HISTORY_KEY)) || [];

    const updated = [item, ...existing].slice(0, 100);

    await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
  } catch (err) {
    console.log("SAVE HISTORY ERROR:", err);
  }
};

export default function App() {
  const [video, setVideo] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [visualsLoading, setVisualsLoading] = useState(false);
  const [selectedZone, setSelectedZone] = useState(null);
  const [pendingVideos, setPendingVideos] = useState([]);

  useEffect(() => {
    loadPendingVideos(setPendingVideos);
  }, []);

  const reset = () => {
    setResult(null);
    setSelectedZone(null);
    setVisualsLoading(false);
  };

  const recordWithCamera = async () => {
    reset();

    const permission = await ImagePicker.requestCameraPermissionsAsync();

    if (!permission.granted) {
      Alert.alert("Camera permission required");
      return;
    }

    const res = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 1,
      videoMaxDuration: 60,
    });

    if (!res.canceled) {
      const a = res.assets[0];

      const selected = {
        uri: a.uri,
        file: a.file,
        name: a.fileName || a.name || "library.mov",
        type: a.mimeType || a.type || "video/quicktime",
      };

      setVideo(selected);

      await savePendingVideo(selected);
      await loadPendingVideos(setPendingVideos);
    }
  };

  const pickWebVideoFile = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "video/*";

    input.onchange = (e) => {
      const file = e.target.files?.[0];

      if (!file) return;

      setVideo({
        file,
        uri: URL.createObjectURL(file),
        name: file.name || "upload.mov",
        type: file.type || "video/mp4",
      });
    };

    input.click();
  };

  const pickFromLibrary = async () => {
    reset();

    if (Platform.OS === "web") {
      pickWebVideoFile();
      return;
    }

    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permission.granted) {
      Alert.alert("Library permission required");
      return;
    }

    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 1,
    });

    if (!res.canceled) {
      const a = res.assets[0];

      const selected = {
        uri: a.uri,
        name: a.fileName || a.name || "library.mov",
        type: a.mimeType || a.type || "video/quicktime",
      };

      setVideo(selected);

      await savePendingVideo(selected);
      await loadPendingVideos(setPendingVideos);
    }
  };

  const pickFromCloud = async () => {
    reset();

    if (Platform.OS === "web") {
      pickWebVideoFile();
      return;
    }

    const res = await DocumentPicker.getDocumentAsync({
      type: "video/*",
      copyToCacheDirectory: true,
    });

    if (!res.canceled) {
      const a = res.assets[0];

      const selected = {
        uri: a.uri,
        name: a.name || "cloud.mov",
        type: a.mimeType || "video/quicktime",
      };

      setVideo(selected);

      await savePendingVideo(selected);
      await loadPendingVideos(setPendingVideos);
    }
  };

  const buildFormData = async (extra = {}) => {
    const formData = new FormData();

    if (Platform.OS === "web") {
      if (!video?.file) {
        throw new Error(
          "No browser file found. Please choose the video again.",
        );
      }

      formData.append("file", video.file, video.name || "upload.mov");
    } else {
      formData.append("file", {
        uri: video.uri,
        name: video.name || "upload.mov",
        type: video.type || "video/mp4",
      });
    }

    Object.entries(extra).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        formData.append(key, value);
      }
    });

    return formData;
  };

  const generateVisuals = async (analysisResult) => {
    if (visualsLoading) return;
    if (result?.phase_images) return;

    try {
      console.log("GENERATE VISUALS STARTED");

      setVisualsLoading(true);

      const bestRep = getBestRep(analysisResult?.rep_feedback || []);

      const visualsRes = await fetch(`${API_URL}/generate_visuals`, {
        method: "POST",
        body: await buildFormData({
          rep_json: bestRep ? JSON.stringify(bestRep) : null,
          exercise_label: analysisResult?.exercise_label || "",
        }),
      });

      const visualsText = await visualsRes.text();

      let visualsData = {};
      try {
        visualsData = JSON.parse(visualsText);
      } catch {
        visualsData = {
          message:
            visualsText || "Visual generation returned non-JSON response",
        };
      }

      console.log("VISUALS RESPONSE:", visualsData);
      console.log("VISUALS STATUS:", visualsRes.status);

      if (!visualsRes.ok) {
        throw new Error(
          visualsData.detail ||
            visualsData.message ||
            "Visual generation request failed",
        );
      }

      setOverlayUrl(fullUrl(visualsData.overlay_video_url));

      setResult((prev) => ({
        ...prev,
        overlay_video_url: visualsData.overlay_video_url,
        phase_images: visualsData.phase_images,
        visuals_error: visualsData.visuals_error,
      }));
    } catch (err) {
      console.log("VISUALS ERROR:", err);

      setResult((prev) => ({
        ...prev,
        visuals_error: err.message,
      }));
    } finally {
      setVisualsLoading(false);
    }
  };

  const [overlayUrl, setOverlayUrl] = useState(null);
  const [overlayLoading, setOverlayLoading] = useState(false);
  const [overlayProgress, setOverlayProgress] = useState("");

  const generateOverlay = async () => {
    try {
      setOverlayLoading(true);
      setOverlayProgress("Generating overlay...");

      const bestRep = getBestRep(result?.rep_feedback || []);

      const formPayload = {
        rep_json: bestRep ? JSON.stringify(bestRep) : null,
        exercise_label: result?.exercise_label || "",
      };

      // 1. Try fast direct overlay first
      try {
        const overlayRes = await fetch(`${API_URL}/generate_overlay`, {
          method: "POST",
          body: await buildFormData(formPayload),
        });

        const overlayData = await overlayRes.json();

        if (overlayRes.ok && overlayData.overlay_video_url) {
          setOverlayProgress("Overlay ready!");
          setOverlayUrl(fullUrl(overlayData.overlay_video_url));

          setResult((prev) => ({
            ...prev,
            overlay_video_url: overlayData.overlay_video_url,
            overlay_error: null,
          }));

          return;
        }
      } catch (directErr) {
        console.log("DIRECT OVERLAY FAILED, FALLING BACK:", directErr);
      }

      // 2. Fallback to background overlay job
      setOverlayProgress("Starting background overlay job...");

      const startRes = await fetch(`${API_URL}/start_overlay`, {
        method: "POST",
        body: await buildFormData(formPayload),
      });

      const startData = await startRes.json();

      if (!startRes.ok || !startData.job_id) {
        throw new Error(startData.message || "Could not start overlay job");
      }

      const jobId = startData.job_id;

      for (let i = 0; i < 40; i++) {
        setOverlayProgress(`Processing overlay... ${i + 1}/40`);

        await new Promise((resolve) => setTimeout(resolve, 3000));

        const statusRes = await fetch(`${API_URL}/overlay_status/${jobId}`);
        const statusData = await statusRes.json();

        if (statusData.status === "ready") {
          setOverlayProgress("Overlay ready!");
          setOverlayUrl(fullUrl(statusData.overlay_video_url));

          setResult((prev) => ({
            ...prev,
            overlay_video_url: statusData.overlay_video_url,
            overlay_error: null,
          }));

          return;
        }

        if (statusData.status === "error") {
          throw new Error(statusData.message || "Overlay generation failed");
        }
      }

      throw new Error("Overlay is still processing. Try again shortly.");
    } catch (err) {
      setOverlayProgress("");

      setResult((prev) => ({
        ...prev,
        overlay_error: err.message,
      }));
    } finally {
      setOverlayLoading(false);
    }
  };

  const analyzeVideo = async () => {
    if (!video) {
      Alert.alert("Pick a video first");
      return;
    }

    setLoading(true);
    setResult(null);
    setSelectedZone(null);
    setVisualsLoading(false);

    try {
      const res = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: await buildFormData(),
      });

      const data = await res.json().catch(() => null);

      if (!data) {
        throw new Error("Server returned an empty response.");
      }

      console.log("ANALYZE RESPONSE:", data);
      console.log("STATUS:", res.status);

      if (!res.ok) {
        throw new Error(
          data?.detail ||
            data?.message ||
            JSON.stringify(data) ||
            "Analyze request failed",
        );
      }

      await saveAnalysisHistory(data);

      setResult(data);
      setLoading(false);

      // Visuals are now generated manually
    } catch (err) {
      setResult({ error: true, message: err.message });
      setLoading(false);
      setVisualsLoading(false);
    }
  };

  const reps = result?.rep_feedback || [];

  const avgScore = useMemo(() => {
    if (!reps.length) return null;

    const avg =
      reps.reduce((sum, rep) => sum + Number(rep.score || 0), 0) / reps.length;

    return Number(avg.toFixed(1));
  }, [reps]);

  const displayScore = avgScore !== null ? Math.round(avgScore * 10) : null;
  const bestRep = getBestRep(reps);

  const biggestFix =
    result?.set_summary?.biggest_fix ||
    bestRep?.feedback?.[0] ||
    "Upload a clear side-angle video for analysis.";

  const phaseConfig = getPhaseConfig(result?.exercise_label);
  const phaseImages = result?.phase_images || {};
  const zones = getInteractiveZones(result);
  const activeZone = selectedZone || zones[0];

  const activeImagePath =
    phaseImages?.[activeZone?.imageKey] ||
    phaseImages?.setup ||
    phaseImages?.bottom ||
    phaseImages?.pull ||
    phaseImages?.mid ||
    phaseImages?.press ||
    phaseImages?.lockout ||
    null;

  const activeImageUrl = fullUrl(activeImagePath);
  const overlayPlayer = useVideoPlayer(overlayUrl, (player) => {
    player.loop = false;
  });

  const buildCoachSummary = (result) => {
    if (!result) return "";

    const rep = result.rep_feedback?.[0];
    if (!rep) return "";

    const issues = rep.issues || [];
    const biggestFix = result.set_summary?.biggest_fix;
    const breakdown = rep.breakdown || {};

    let intro = "Solid rep overall.";
    let body = [];
    let cues = [];

    // Tone based on score
    if (rep.score >= 9) {
      intro = "Great rep — very strong execution.";
    } else if (rep.score >= 7) {
      intro = "Good rep overall, but there are a couple things to clean up.";
    } else {
      intro = "This rep needs some work.";
    }

    // Explain issue
    if (issues.length > 0) {
      body.push(issues[0]);
    }

    // Smart cues
    if (breakdown.knees === "poor") {
      cues.push("Drive your knees out and keep them tracking over your toes.");
    }

    if (breakdown.depth === "borderline") {
      cues.push("Sit slightly deeper while keeping your chest up.");
    }

    if (breakdown.torso === "poor") {
      cues.push("Keep your chest tall and avoid leaning forward.");
    }

    if (breakdown.heels === "poor") {
      cues.push("Keep your weight through your mid-foot and heels.");
    }

    // Build paragraph
    let text = intro;

    if (body.length > 0) {
      text += " " + body.join(" ");
    }

    if (cues.length > 0) {
      text += " Focus on this next: " + cues.join(" ");
    }

    return text;
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.hero}>
          <View>
            <Text style={styles.eyebrow}>AI Movement Coach</Text>
            <Text style={styles.title}>FormCheck AI</Text>
            <Text style={styles.subtitle}>
              Upload a lift. Get scoring, coaching zones, phase images, and
              replay.
            </Text>
          </View>

          <View style={styles.logoBubble}>
            <Text style={styles.logoText}>AI</Text>
          </View>
        </View>

        <View style={styles.actionRow}>
          <TouchableOpacity
            style={styles.primaryButton}
            onPress={recordWithCamera}
          >
            <Text style={styles.primaryButtonText}>Record</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.secondaryButton}
            onPress={pickFromLibrary}
          >
            <Text style={styles.secondaryButtonText}>Library</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.secondaryButton}
            onPress={pickFromCloud}
          >
            <Text style={styles.secondaryButtonText}>Files</Text>
          </TouchableOpacity>
        </View>

        {video && (
          <View style={styles.selectedCard}>
            <Text style={styles.cardLabel}>Selected Video</Text>
            <Text style={styles.selectedName}>{video.name}</Text>
          </View>
        )}

        {pendingVideos.length > 0 && (
          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Pending Videos</Text>
            <Text style={styles.sectionSub}>
              Saved locally for later analysis
            </Text>

            {pendingVideos.map((v) => (
              <TouchableOpacity
                key={v.id}
                style={styles.selectedCard}
                onPress={() => setVideo(v)}
              >
                <Text style={styles.cardLabel}>Pending</Text>
                <Text style={styles.selectedName}>{v.name}</Text>
              </TouchableOpacity>
            ))}
          </View>
        )}

        <TouchableOpacity
          style={[
            styles.analyzeButton,
            (loading || visualsLoading) && styles.disabledButton,
          ]}
          onPress={analyzeVideo}
          disabled={loading || visualsLoading}
        >
          {loading ? (
            <View style={styles.loadingBlock}>
              <ActivityIndicator color="#020617" />
              <Text style={styles.analyzeButtonText}>
                Analyzing movement...
              </Text>
            </View>
          ) : visualsLoading ? (
            <View style={styles.loadingBlock}>
              <ActivityIndicator color="#020617" />
              <Text style={styles.analyzeButtonText}>Building visuals...</Text>
            </View>
          ) : (
            <Text style={styles.analyzeButtonText}>Analyze Lift</Text>
          )}
        </TouchableOpacity>

        {result?.error && (
          <View style={styles.errorCard}>
            <Text style={styles.errorTitle}>Request Failed</Text>
            <Text style={styles.errorText}>{result.message}</Text>
          </View>
        )}

        {result?.feedback && !result.error && reps.length === 0 && (
          <View style={styles.errorCard}>
            <Text style={styles.errorTitle}>Video Not Usable</Text>
            <Text style={styles.errorText}>
              Camera angle is too close or unclear for reliable scoring.
            </Text>
            <Text style={styles.errorText}>
              Move farther back, record from the side, and keep the full body
              visible.
            </Text>
          </View>
        )}

        {result && !result.error && reps.length > 0 && (
          <>
            <View style={styles.dashboardGrid}>
              <View style={styles.scoreCard}>
                <Text style={styles.cardLabel}>Overall Score</Text>

                <View
                  style={[
                    styles.scoreCircle,
                    { borderColor: getScoreColor(avgScore || 0) },
                  ]}
                >
                  <Text style={styles.scoreBig}>{displayScore}</Text>
                  <Text style={styles.scoreSmall}>/100</Text>
                </View>

                <Text style={styles.exerciseName}>{result.exercise_label}</Text>
                <Text style={styles.confidenceText}>
                  Confidence {Math.round(Number(result.confidence || 0) * 100)}%
                </Text>
              </View>

              <View style={styles.insightCard}>
                <Text style={styles.cardLabel}>Biggest Fix</Text>
                <Text style={styles.bigFix}>{biggestFix}</Text>

                <View style={styles.miniStats}>
                  <View style={styles.statPill}>
                    <Text style={styles.statNumber}>
                      {result?.set_summary?.detected_reps || reps.length}
                    </Text>
                    <Text style={styles.statLabel}>Reps</Text>
                  </View>

                  <View style={styles.statPill}>
                    <Text style={styles.statNumber}>
                      {result?.set_summary?.best_rep || bestRep?.rep || "-"}
                    </Text>
                    <Text style={styles.statLabel}>Best</Text>
                  </View>

                  <View style={styles.statPill}>
                    <Text style={styles.statNumber}>
                      {result?.set_summary?.worst_rep || "-"}
                    </Text>
                    <Text style={styles.statLabel}>Needs Work</Text>
                  </View>
                </View>
              </View>
            </View>

            {visualsLoading && (
              <View style={styles.warningCard}>
                <ActivityIndicator color="#86efac" />
                <Text style={styles.warningTitle}>
                  Coaching visuals are loading
                </Text>
                <Text style={styles.warningText}>
                  Scores are ready. Phase images and replay will appear next.
                </Text>
              </View>
            )}

            {result?.visuals_error && (
              <View style={styles.errorCard}>
                <Text style={styles.errorTitle}>Visuals Could Not Load</Text>
                <Text style={styles.errorText}>{result.visuals_error}</Text>
              </View>
            )}

            {
              <TouchableOpacity
                style={[
                  styles.analyzeButton,
                  (visualsLoading || result?.phase_images) &&
                    styles.disabledButton,
                ]}
                onPress={() => generateVisuals(result)}
                disabled={visualsLoading || !!result?.phase_images}
              >
                <Text style={styles.analyzeButtonText}>
                  {visualsLoading
                    ? "Generating Phase Review..."
                    : result?.phase_images
                      ? "Phase Review Generated"
                      : "Generate Phase Review"}
                </Text>
              </TouchableOpacity>
            }

            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Rep Breakdown</Text>
              <Text style={styles.sectionSub}>Score trend across the set</Text>
              {reps.map((rep) => {
                const score = Number(rep.score || 0);
                const barWidth = `${Math.min(100, Math.max(8, score * 10))}%`;

                return (
                  <View key={`rep-${rep.rep}`} style={styles.repRow}>
                    <View style={styles.repTop}>
                      <Text style={styles.repLabel}>Rep {rep.rep}</Text>
                      <Text style={styles.repScore}>{score.toFixed(1)}/10</Text>
                    </View>

                    <View style={styles.repBarTrack}>
                      <View
                        style={[
                          styles.repBarFill,
                          {
                            width: barWidth,
                            backgroundColor: getScoreColor(score),
                          },
                        ]}
                      />
                    </View>

                    <Text style={styles.repFeedback}>
                      {rep.feedback?.[0] || rep.issues?.[0] || "Good rep."}
                    </Text>
                  </View>
                );
              })}
            </View>

            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Interactive Coaching Map</Text>
              <Text style={styles.sectionSub}>
                Tap a zone to see the most relevant saved phase frame.
              </Text>

              <View style={styles.coachImageWrap}>
                {activeImageUrl ? (
                  <Image
                    key={`${activeZone?.id}-${activeImagePath}`}
                    source={{ uri: activeImageUrl }}
                    style={styles.coachImage}
                                          resizeMode={
                        result?.exercise_label?.toLowerCase().includes("muscle") ||
                        result?.exercise_label?.toLowerCase().includes("pull")
                          ? "contain"
                          : "cover"
                      }
                  />
                ) : (
                  <View style={styles.emptyImage}>
                    <Text style={styles.emptyImageText}>
                      Visuals are not available yet.
                    </Text>
                  </View>
                )}
              </View>

              <View style={styles.zoneGrid}>
                {zones.map((zone) => {
                  const isActive = activeZone?.id === zone.id;
                  const color = getStatusColor(zone.status);

                  return (
                    <TouchableOpacity
                      key={zone.id}
                      style={[
                        styles.zonePill,
                        isActive && styles.zonePillActive,
                        isActive && { borderColor: color },
                      ]}
                      onPress={() => setSelectedZone(zone)}
                    >
                      <View
                        style={[styles.statusDot, { backgroundColor: color }]}
                      />
                      <Text style={styles.zoneText}>{zone.title}</Text>
                    </TouchableOpacity>
                  );
                })}
              </View>

              {activeZone && (
                <View style={styles.coachingNote}>
                  <Text style={styles.noteTitle}>{activeZone.title}</Text>
                  <Text style={styles.noteStatus}>
                    Status: {formatLabel(activeZone.status)}
                  </Text>
                  <Text style={styles.noteText}>{activeZone.note}</Text>
                </View>
              )}
            </View>

            <View style={styles.card}>
              <Text style={styles.sectionTitle}>{phaseConfig.title}</Text>
              <Text style={styles.sectionSub}>{phaseConfig.text}</Text>

              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={styles.phaseScroller}
              >
                {phaseConfig.items.map(([key, label]) => {
                  const path = phaseImages[key];
                  const url = fullUrl(path);

                  return (
                    <View key={key} style={styles.phaseCard}>
                      <View style={styles.phaseImageWrap}>
                        {url ? (
                          <Image
                            source={{ uri: url }}
                            style={styles.phaseImage}
                                                          resizeMode={
                                result?.exercise_label
                                  ?.toLowerCase()
                                  .includes("muscle") ||
                                result?.exercise_label
                                  ?.toLowerCase()
                                  .includes("pull")
                                  ? "contain"
                                  : "cover"
                              }
                          />
                        ) : (
                          <View style={styles.emptyPhase}>
                            <Text style={styles.emptyPhaseText}>No image</Text>
                          </View>
                        )}
                      </View>
                      <Text style={styles.phaseLabel}>{label}</Text>
                    </View>
                  );
                })}
              </ScrollView>
            </View>

            <TouchableOpacity
              style={[
                styles.overlayButton,
                overlayLoading && styles.disabledButton,
              ]}
              onPress={generateOverlay}
              disabled={overlayLoading}
            >
              {overlayLoading ? (
                <View style={styles.loadingBlock}>
                  <ActivityIndicator color="#020617" />
                  <Text style={styles.analyzeButtonText}>
                    {overlayProgress || "Generating overlay..."}
                  </Text>
                </View>
              ) : (
                <Text style={styles.analyzeButtonText}>Generate Overlay</Text>
              )}
            </TouchableOpacity>

            {overlayUrl && (
              <View style={styles.overlayCard}>
                <Text style={styles.sectionTitle}>Coached Replay</Text>
                <Text style={styles.sectionSub}>
                  Overlay video with rep feedback and movement markers.
                </Text>

                <VideoView
                  player={overlayPlayer}
                  style={styles.videoPlayer}
                  allowsFullscreen
                  allowsPictureInPicture
                  nativeControls
                  contentFit="contain"
                />
              </View>
            )}

            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Coach Summary</Text>

              <Text style={styles.coachText}>{buildCoachSummary(result)}</Text>
            </View>
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#020617",
  },
  content: {
    padding: 18,
    paddingBottom: 42,
  },
  hero: {
    backgroundColor: "#0f172a",
    borderRadius: 28,
    padding: 22,
    borderWidth: 1,
    borderColor: "#1e293b",
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 16,
    marginBottom: 16,
  },
  eyebrow: {
    color: "#86efac",
    fontSize: 13,
    fontWeight: "800",
    letterSpacing: 1,
    textTransform: "uppercase",
    marginBottom: 8,
  },
  title: {
    color: "#f8fafc",
    fontSize: 34,
    fontWeight: "900",
    letterSpacing: -1,
  },
  subtitle: {
    color: "#94a3b8",
    fontSize: 15,
    lineHeight: 22,
    marginTop: 8,
    maxWidth: 280,
  },
  logoBubble: {
    width: 58,
    height: 58,
    borderRadius: 20,
    backgroundColor: "#86efac",
    alignItems: "center",
    justifyContent: "center",
  },
  logoText: {
    color: "#020617",
    fontSize: 20,
    fontWeight: "900",
  },
  actionRow: {
    flexDirection: "row",
    gap: 10,
    marginBottom: 14,
  },
  primaryButton: {
    flex: 1,
    backgroundColor: "#22c55e",
    borderRadius: 18,
    paddingVertical: 15,
    alignItems: "center",
  },
  primaryButtonText: {
    color: "#020617",
    fontWeight: "900",
    fontSize: 15,
  },
  secondaryButton: {
    flex: 1,
    backgroundColor: "#111827",
    borderRadius: 18,
    paddingVertical: 15,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#1f2937",
  },
  secondaryButtonText: {
    color: "#e5e7eb",
    fontWeight: "800",
    fontSize: 15,
  },
  selectedCard: {
    backgroundColor: "#0f172a",
    borderRadius: 22,
    padding: 16,
    borderWidth: 1,
    borderColor: "#1e293b",
    marginBottom: 14,
  },
  cardLabel: {
    color: "#64748b",
    fontSize: 12,
    fontWeight: "900",
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  selectedName: {
    color: "#f8fafc",
    fontSize: 15,
    fontWeight: "700",
    marginTop: 6,
  },
  analyzeButton: {
    backgroundColor: "#86efac",
    borderRadius: 22,
    paddingVertical: 18,
    alignItems: "center",
    marginBottom: 18,
  },
  disabledButton: {
    opacity: 0.75,
  },
  analyzeButtonText: {
    color: "#020617",
    fontSize: 17,
    fontWeight: "900",
  },
  loadingBlock: {
    alignItems: "center",
    gap: 8,
  },
  dashboardGrid: {
    gap: 14,
    marginBottom: 14,
  },
  scoreCard: {
    backgroundColor: "#0f172a",
    borderRadius: 28,
    padding: 22,
    borderWidth: 1,
    borderColor: "#1e293b",
    alignItems: "center",
  },
  scoreCircle: {
    width: 154,
    height: 154,
    borderRadius: 77,
    borderWidth: 12,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 18,
    marginBottom: 14,
    backgroundColor: "#020617",
  },
  scoreBig: {
    color: "#f8fafc",
    fontSize: 42,
    fontWeight: "900",
    lineHeight: 48,
  },
  scoreSmall: {
    color: "#94a3b8",
    fontWeight: "800",
  },
  exerciseName: {
    color: "#f8fafc",
    fontSize: 23,
    fontWeight: "900",
    marginTop: 4,
  },
  confidenceText: {
    color: "#94a3b8",
    marginTop: 6,
    fontWeight: "700",
  },
  insightCard: {
    backgroundColor: "#0f172a",
    borderRadius: 28,
    padding: 20,
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  bigFix: {
    color: "#f8fafc",
    fontSize: 22,
    fontWeight: "900",
    lineHeight: 29,
    marginTop: 10,
  },
  miniStats: {
    flexDirection: "row",
    gap: 10,
    marginTop: 18,
  },
  statPill: {
    flex: 1,
    backgroundColor: "#020617",
    borderRadius: 18,
    paddingVertical: 14,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  statNumber: {
    color: "#86efac",
    fontSize: 22,
    fontWeight: "900",
  },
  statLabel: {
    color: "#94a3b8",
    fontSize: 11,
    fontWeight: "800",
    marginTop: 3,
    textTransform: "uppercase",
  },
  card: {
    backgroundColor: "#0f172a",
    borderRadius: 28,
    padding: 18,
    borderWidth: 1,
    borderColor: "#1e293b",
    marginBottom: 14,
  },
  sectionTitle: {
    color: "#f8fafc",
    fontSize: 21,
    fontWeight: "900",
  },
  sectionSub: {
    color: "#94a3b8",
    fontSize: 13,
    fontWeight: "700",
    marginTop: 5,
    marginBottom: 14,
  },
  repRow: {
    marginTop: 14,
  },
  repTop: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 8,
  },
  repLabel: {
    color: "#e5e7eb",
    fontWeight: "900",
  },
  repScore: {
    color: "#f8fafc",
    fontWeight: "900",
  },
  repBarTrack: {
    height: 12,
    borderRadius: 999,
    backgroundColor: "#020617",
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  repBarFill: {
    height: "100%",
    borderRadius: 999,
  },
  repFeedback: {
    color: "#94a3b8",
    fontSize: 13,
    marginTop: 7,
    lineHeight: 18,
  },
  coachImageWrap: {
    height: 260,
    borderRadius: 22,
    overflow: "hidden",
    backgroundColor: "#020617",
    borderWidth: 1,
    borderColor: "#1e293b",
    marginBottom: 14,
  },
  coachImage: {
    width: "100%",
    height: "100%",
  },
  emptyImage: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
  },
  emptyImageText: {
    color: "#64748b",
    fontWeight: "800",
    textAlign: "center",
  },
  zoneGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  zonePill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: 13,
    paddingVertical: 11,
    borderRadius: 999,
    backgroundColor: "#020617",
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  zonePillActive: {
    backgroundColor: "#111827",
    borderWidth: 2,
  },
  statusDot: {
    width: 9,
    height: 9,
    borderRadius: 999,
  },
  zoneText: {
    color: "#e5e7eb",
    fontWeight: "800",
    fontSize: 13,
  },
  coachingNote: {
    marginTop: 14,
    backgroundColor: "#020617",
    borderRadius: 20,
    padding: 16,
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  noteTitle: {
    color: "#f8fafc",
    fontSize: 18,
    fontWeight: "900",
  },
  noteStatus: {
    color: "#86efac",
    fontWeight: "900",
    marginTop: 6,
  },
  noteText: {
    color: "#cbd5e1",
    fontSize: 14,
    lineHeight: 21,
    marginTop: 8,
  },
  phaseScroller: {
    gap: 12,
    paddingRight: 12,
  },
  phaseCard: {
    width: 162,
    backgroundColor: "#020617",
    borderRadius: 20,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  phaseImageWrap: {
    height: 128,
    backgroundColor: "#111827",
  },
  phaseImage: {
    width: "100%",
    height: "100%",
  },
  emptyPhase: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  emptyPhaseText: {
    color: "#64748b",
    fontWeight: "800",
  },
  phaseLabel: {
    color: "#f8fafc",
    fontWeight: "900",
    padding: 12,
    textAlign: "center",
  },
  warningCard: {
    backgroundColor: "#102018",
    borderRadius: 24,
    padding: 18,
    borderWidth: 1,
    borderColor: "#14532d",
    marginBottom: 14,
    gap: 8,
  },
  warningTitle: {
    color: "#bbf7d0",
    fontSize: 16,
    fontWeight: "900",
  },
  warningText: {
    color: "#dcfce7",
    lineHeight: 20,
  },
  errorCard: {
    backgroundColor: "#2a1111",
    borderRadius: 24,
    padding: 18,
    borderWidth: 1,
    borderColor: "#7f1d1d",
    marginBottom: 14,
  },
  errorTitle: {
    color: "#fecaca",
    fontSize: 17,
    fontWeight: "900",
    marginBottom: 8,
  },
  errorText: {
    color: "#fecaca",
    lineHeight: 21,
  },
  summaryLine: {
    color: "#e5e7eb",
    fontSize: 15,
    lineHeight: 22,
    marginTop: 10,
    marginBottom: 10,
  },
  feedbackLine: {
    color: "#94a3b8",
    lineHeight: 22,
    marginTop: 4,
  },
  overlayButton: {
    backgroundColor: "#22c55e",
    borderRadius: 18,
    paddingVertical: 16,
    alignItems: "center",
    marginBottom: 14,
  },
  overlayButtonDisabled: {
    backgroundColor: "#374151",
    borderRadius: 18,
    paddingVertical: 16,
    alignItems: "center",
    marginBottom: 14,
    opacity: 0.7,
  },
  overlayCard: {
    backgroundColor: "#111827",
    borderRadius: 28,
    padding: 18,
    marginBottom: 18,
    borderWidth: 1,
    borderColor: "#243044",
    overflow: "visible",
  },
  videoPlayer: {
    width: "100%",
    height: 620,
    borderRadius: 22,
    backgroundColor: "#020617",
  },
  videoShell: {
    width: "100%",
    borderRadius: 22,
    overflow: "visible",
    backgroundColor: "#020617",
  },
  coachText: {
    color: "#E5E7EB",
    fontSize: 16,
    lineHeight: 22,
    marginTop: 8,
  },
});
